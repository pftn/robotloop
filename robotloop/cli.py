"""RobotLoop 命令行入口 —— 数据闭环全流程。

    robotloop ingest-demo --store ./lake -n 60          # 合成数据灌库（离线可跑）
    robotloop ingest-mcap --bag x.mcap --camera /cam --state /joints \\
        --task pick_red_cube --embodiment aloha --store ./lake
    robotloop search --store ./lake --query "找所有 ALOHA 双臂成功抓取红色方块的轨迹"
    robotloop quality --store ./lake --report ./quality.html
    robotloop export --store ./lake --out ./ft_data --version v2.1 \\
        --filters '{"embodiment_tag":"aloha"}' --success-only --with-modality
    robotloop finetune-script --dataset lerobot/pusht --out ./run_act.sh
"""

from __future__ import annotations

import argparse
import json
import sys


def _open_store(path: str):
    """打开数据湖：本地路径 → LocalStore；iceberg://[表名] → 生产湖
    （Iceberg REST catalog + MinIO，连接参数走 ICEBERG_CATALOG_URI /
    S3_ENDPOINT / S3_ACCESS_KEY / S3_SECRET_KEY 平台环境变量）。"""
    if path.startswith("iceberg://"):
        from robotloop.retrieval.store import MilvusIcebergStore

        table = path[len("iceberg://") :].strip("/") or "robotloop.episodes"
        return MilvusIcebergStore(episodes_table=table)
    from robotloop.retrieval.store import LocalStore

    return LocalStore(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="robotloop", description="RobotLoop 具身数据闭环 CLI"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest-demo", help="合成演示数据灌库")
    p.add_argument("--store", required=True)
    p.add_argument("-n", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("ingest-mcap", help="MCAP/rosbag2 灌库")
    p.add_argument("--bag", required=True)
    p.add_argument("--camera", action="append", default=[])
    p.add_argument("--state", required=True)
    p.add_argument("--action-topic", default=None)
    p.add_argument("--task", required=True)
    p.add_argument("--instruction", default="")
    p.add_argument("--embodiment", required=True)
    p.add_argument("--source", default="teleop", choices=["teleop", "sim", "real"])
    p.add_argument("--success", choices=["true", "false", "unknown"], default="unknown")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--tolerance-ms", type=float, default=20.0)
    p.add_argument("--store", required=True)

    p = sub.add_parser("search", help="混合检索")
    p.add_argument("--store", required=True)
    p.add_argument("--query", default=None, help="自然语言查询（自动抽结构化谓词）")
    p.add_argument("--filters", default=None, help="JSON 结构化过滤")
    p.add_argument("--top-k", type=int, default=10)

    p = sub.add_parser("quality", help="质检报告")
    p.add_argument("--store", required=True)
    p.add_argument("--report", default=None, help="HTML 报告输出路径")
    p.add_argument("--dedup-threshold", type=float, default=0.98)

    p = sub.add_parser("export", help="导出 LeRobot 训练数据集")
    p.add_argument(
        "--store",
        required=True,
        help="本地湖路径，或 iceberg://robotloop.episodes 生产湖"
        "（配 ICEBERG_CATALOG_URI=http://localhost:8181 "
        "S3_ENDPOINT=http://localhost:9000 等环境变量）",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--version", default="v2.1", choices=["v2.1", "v3.0"])
    p.add_argument("--filters", default=None)
    p.add_argument("--success-only", action="store_true")
    p.add_argument(
        "--with-modality", action="store_true", help="写 GR00T modality.json"
    )
    p.add_argument("--fps", type=float, default=None)
    p.add_argument(
        "--video-size",
        default=None,
        help="显式帧尺寸 HxW（如 480x640，对齐 aloha 等仿真 env 渲染分辨率）",
    )
    p.add_argument(
        "--camera-map",
        default=None,
        help='相机键名映射 JSON（如 {"/cam/image": "observation.images.top"}，'
        "对齐仿真 env 相机键，缺省按 topic 名自动映射）",
    )

    p = sub.add_parser(
        "finetune-script", help="生成 LeRobot ACT/Diffusion 训练脚本（AutoDL 用）"
    )
    p.add_argument(
        "--dataset", default="lerobot/pusht", help="repo_id 或 AutoDL 上的本地路径"
    )
    p.add_argument("--out", required=True)
    p.add_argument("--policy", default="act", choices=["act", "diffusion"])
    p.add_argument(
        "--env",
        default="pusht",
        help="仿真评测环境: pusht | aloha；）",
    )
    p.add_argument("--max-steps", type=int, default=20000)

    args = parser.parse_args(argv)

    if args.cmd == "ingest-demo":
        from robotloop.datasets.demo import make_demo_episodes
        from robotloop.datasets.load import ingest_episodes

        store = _open_store(args.store)
        eps = make_demo_episodes(n=args.n, seed=args.seed)
        res = ingest_episodes(eps, store)
        print(f"✔ 灌库完成: {json.dumps(res, ensure_ascii=False)}")

    elif args.cmd == "ingest-mcap":
        from robotloop.datasets.load import ingest_episodes
        from robotloop.pipeline import bag_to_episode
        from robotloop.schema.episode import DataSource

        ep = bag_to_episode(
            args.bag,
            camera_topics=args.camera,
            state_topic=args.state,
            action_topic=args.action_topic,
            task=args.task,
            language_instruction=args.instruction or args.task,
            embodiment_tag=args.embodiment,
            source=DataSource(args.source),
            success={"true": True, "false": False, "unknown": None}[args.success],
            target_fps=args.fps,
            tolerance=args.tolerance_ms / 1000.0,
        )
        store = _open_store(args.store)
        res = ingest_episodes([ep], store)
        print(
            f"✔ MCAP 灌库: {ep.num_frames} frames, 对齐报告: {json.dumps(ep.metadata['alignment_report'], ensure_ascii=False)}"
        )

    elif args.cmd == "search":
        from robotloop.retrieval.hybrid import HybridRetriever

        store = _open_store(args.store)
        retriever = HybridRetriever(store)
        filters = json.loads(args.filters) if args.filters else None
        res = retriever.search(
            text=args.query,
            filters=filters,
            top_k=args.top_k,
            nl_query=bool(args.query),
        )
        print(json.dumps(res, ensure_ascii=False, indent=2, default=str))

    elif args.cmd == "quality":
        import numpy as np

        from robotloop.quality.dashboard import render_html_report, task_distribution
        from robotloop.quality.dedup import dedup_by_similarity

        store = _open_store(args.store)
        stats = task_distribution(store.all_meta())
        extra = {}
        ids, _text, traj = (
            store.embeddings() if hasattr(store, "embeddings") else ([], None, None)
        )
        if ids and traj is not None and traj.size:
            dd = dedup_by_similarity(ids, traj, threshold=args.dedup_threshold)
            extra["相似去重"] = dd.summary
        print(json.dumps({"stats": stats, **extra}, ensure_ascii=False, indent=2))
        if args.report:
            render_html_report(stats, args.report, extra_sections=extra)
            print(f"✔ HTML 报告: {args.report}")

    elif args.cmd == "export":
        from robotloop.export.lerobot_export import export_to_lerobot

        store = _open_store(args.store)
        filters = json.loads(args.filters) if args.filters else None
        video_size = None
        if args.video_size:
            h, w = args.video_size.lower().split("x")
            video_size = (int(h), int(w))
        camera_map = json.loads(args.camera_map) if args.camera_map else None
        info = export_to_lerobot(
            store,
            args.out,
            filters=filters,
            version=args.version,
            success_only=args.success_only,
            fps=args.fps,
            video_size=video_size,
            camera_map=camera_map,
        )
        if args.with_modality:
            from robotloop.export.gr00t import write_modality_json

            dim = info["features"]["action"]["shape"][0]
            write_modality_json(
                args.out,
                state_groups={"joint_positions": (0, dim)},
                action_groups={"joint_positions": (0, dim)},
            )
            print("✔ 已写入 meta/modality.json（GR00T 兼容）")
        print(
            f"✔ 导出完成: {info['total_episodes']} episodes / {info['total_frames']} frames → {args.out} ({info['codebase_version']})"
        )

    elif args.cmd == "finetune-script":
        from robotloop.export.act_train import render_act_train_script

        render_act_train_script(
            dataset=args.dataset,
            policy=args.policy,
            env=args.env,
            steps=args.max_steps,
            out_path=args.out,
        )
        print(
            f"✔ 训练脚本: {args.out}（scp 到 AutoDL 4090 后 bash 执行；GR00T/OpenVLA 微调架构上支持，留作进阶）"
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
