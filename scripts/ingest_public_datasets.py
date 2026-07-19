#!/usr/bin/env python3
"""公开数据集灌库脚本。

数据集一律按名称指定：--lerobot <repo_id> / --openx <tfds_name>，
推荐清单预设见 config/presets.yaml（--preset <name> 读取）。

下载前对每个数据源做 size preflight（预估大小与 episode 数并打印），
预估超过 10GB 的数据源需要 --yes 确认才会继续。

    元数据 → Iceberg robotloop.episodes（REST catalog）
    帧数据 → MinIO s3://robotloop-data/frames/{episode_id}.parquet
    向量   → Milvus/Zilliz episode_vectors（CLIP 512d）

用法：
    # 内置合成数据（CI / 无集群环境，秒级）
    python scripts/ingest_public_datasets.py --demo --backend local --store-path ./local_store

    # 预设清单（config/presets.yaml）
    python scripts/ingest_public_datasets.py --preset demo --backend production

    # 单数据集
    python scripts/ingest_public_datasets.py \
        --lerobot lerobot/aloha_sim_insertion_human --max-episodes 50 \
        --backend production

    # Open X 子集（TF/TFDS 依赖重，单独 Docker 隔离）
    python scripts/ingest_public_datasets.py --openx viola toto --max-episodes 500

    # preflight 超 10GB 自动确认（CI 场景）
    python scripts/ingest_public_datasets.py --preset demo --yes
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest-public")

SIZE_CONFIRM_GB = 10.0  # 超过此预估大小需 --yes 确认


def load_preset(name: str) -> dict:
    """从 config/presets.yaml 读取预设清单。"""
    import yaml

    path = os.path.join(os.path.dirname(__file__), "..", "config", "presets.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    presets = cfg.get("presets", {})
    if name not in presets:
        raise SystemExit(f"未知预设 {name!r}，可选: {sorted(presets)}（config/presets.yaml）")
    return presets[name]


def preflight(lerobot_repos, openx_names) -> bool:
    """打印各数据源预估大小与 episode 数；返回是否可继续（超 10GB 未确认时 False）。"""
    from robotloop.datasets.lerobot_hub import preflight_lerobot
    from robotloop.datasets.openx import preflight_openx

    oversized = []
    print(f"{'source':<8} {'dataset':<40} {'size(GB)':>10} {'episodes':>10}")
    print("-" * 74)
    for repo in lerobot_repos:
        info = preflight_lerobot(repo)
        size = f"{info['size_gb']:.2f}" if info["size_gb"] is not None else "unknown"
        eps = str(info["episodes"]) if info["episodes"] is not None else "unknown"
        print(f"{'lerobot':<8} {repo:<40} {size:>10} {eps:>10}")
        if info["size_gb"] and info["size_gb"] > SIZE_CONFIRM_GB:
            oversized.append((repo, info["size_gb"]))
    for name in openx_names:
        info = preflight_openx(name)
        size = f"{info['size_gb']:.2f}" if info["size_gb"] is not None else "unknown"
        eps = str(info["episodes"]) if info["episodes"] is not None else "unknown"
        print(f"{'openx':<8} {name:<40} {size:>10} {eps:>10}")
        if info["size_gb"] and info["size_gb"] > SIZE_CONFIRM_GB:
            oversized.append((name, info["size_gb"]))

    if oversized:
        for ds, gb in oversized:
            logger.warning("%s 预估 %.2f GB，超过 %.0f GB 确认线", ds, gb, SIZE_CONFIRM_GB)
        return False
    return True


def build_backend(name: str, store_path: str):
    """构造 (store, episode_sink, milvus_client)。"""
    if name == "local":
        from robotloop.retrieval.store import LocalStore

        return LocalStore(store_path), None, None

    # production：Iceberg REST + MinIO + Zilliz
    from robotloop.retrieval.store import MilvusIcebergStore
    from robotloop.schema.sink import EpisodeSink

    store = MilvusIcebergStore()
    sink = EpisodeSink()
    milvus = store._milvus  # 复用同一连接写向量
    return store, sink, milvus


def main():
    ap = argparse.ArgumentParser(description="公开数据集 → RobotLoop 统一入库")
    ap.add_argument("--backend", choices=["local", "production"], default="local")
    ap.add_argument("--store-path", default="./local_store")
    ap.add_argument("--lerobot", nargs="*", default=[], help="LeRobot Hub repo id 列表")
    ap.add_argument("--openx", nargs="*", default=[], help="Open X TFDS 子集名列表")
    ap.add_argument("--demo", action="store_true", help="灌内置合成 demo 数据集（60 条，CI 用）")
    ap.add_argument("--preset", default=None, help="预设清单名（config/presets.yaml）")
    ap.add_argument("--max-episodes", type=int, default=50, help="每个数据源最多灌多少条")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--yes", action="store_true",
                    help=f"preflight 预估超过 {SIZE_CONFIRM_GB:.0f}GB 时自动确认继续")
    args = ap.parse_args()

    # HF 镜像
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    if args.preset:
        preset = load_preset(args.preset)
        args.lerobot = [d["repo_id"] for d in preset.get("lerobot", [])] + args.lerobot
        args.openx = [d["tfds_name"] for d in preset.get("openx", [])] + args.openx
        args.max_episodes = max(args.max_episodes, int(preset.get("max_episodes_per_source", 50)))
        logger.info("preset=%s -> lerobot=%s openx=%s max_episodes=%d",
                    args.preset, args.lerobot, args.openx, args.max_episodes)

    # size preflight：真实数据源（--demo 合成数据无需 preflight）
    if (args.lerobot or args.openx) and not args.yes:
        if not preflight(args.lerobot, args.openx):
            raise SystemExit(
                f"存在预估超过 {SIZE_CONFIRM_GB:.0f}GB 的数据源，加 --yes 确认后继续"
            )

    from robotloop.datasets.load import ingest_episodes

    store, sink, milvus = build_backend(args.backend, args.store_path)
    logger.info("backend=%s store=%s sink=%s milvus=%s",
                args.backend, type(store).__name__, bool(sink), bool(milvus))

    totals = {"episodes": 0, "steps": 0}

    if args.demo:
        from robotloop.datasets.demo import make_demo_episodes

        eps = make_demo_episodes()[: args.max_episodes] if args.max_episodes else make_demo_episodes()
        r = ingest_episodes(eps, store, batch_size=args.batch_size,
                            episode_sink=sink, milvus_client=milvus)
        totals["episodes"] += r["episodes"]
        totals["steps"] += r["steps"]
        logger.info("demo: %s", r)

    for repo in args.lerobot:
        from robotloop.datasets.lerobot_hub import load_lerobot_hub

        logger.info("loading LeRobot dataset: %s (HF_ENDPOINT=%s)", repo, os.environ["HF_ENDPOINT"])
        eps = load_lerobot_hub(repo)[: args.max_episodes]
        r = ingest_episodes(eps, store, batch_size=args.batch_size,
                            episode_sink=sink, milvus_client=milvus)
        totals["episodes"] += r["episodes"]
        totals["steps"] += r["steps"]
        logger.info("%s: %s", repo, r)

    for name in args.openx:
        from robotloop.datasets.openx import load_openx_subset

        logger.info("loading Open X subset: %s", name)
        eps = load_openx_subset(name, max_episodes=args.max_episodes)
        r = ingest_episodes(eps, store, batch_size=args.batch_size,
                            episode_sink=sink, milvus_client=milvus)
        totals["episodes"] += r["episodes"]
        totals["steps"] += r["steps"]
        logger.info("%s: %s", name, r)

    logger.info("DONE: %d episodes / %d steps, store_count=%d",
                totals["episodes"], totals["steps"], store.count())


if __name__ == "__main__":
    main()
