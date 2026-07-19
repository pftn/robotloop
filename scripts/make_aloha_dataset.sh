#!/bin/bash
# =============================================================================
# 自采 aloha 数据一条龙：造包 → 上传灌库 → 等待入库 → 导出 → 拷出 → 校验
#
# 产出的数据集已通过沙箱端到端验证（lerobot v0.3.3 加载端全项检查），
# 可直接训练并配 --env.type=aloha 做仿真评测。
#
# 前置（一次性）：
#   1. docker compose 全栈已启动（minio / kafka / ray-worker / iceberg-rest）
#   2. 本机 venv 已装：pip install mcap mcap-ros2-support boto3 Pillow
#   3. 在仓库根目录执行本脚本
#
# ⚠️ episode_id 为 uuid 随机生成，重复执行会产生重复 episode。
#    重跑前请先清理旧数据（见脚本末尾 CLEAN 段说明）。
#
# 用法：bash scripts/make_aloha_dataset.sh [N]     # N=造几条包，默认 5
# =============================================================================
set -euo pipefail

N=${1:-5}
OUT_DIR=device/sample_bags
EXPORT_DIR=/tmp/ft_mcap           # 容器内路径
LOCAL_EXPORT=/tmp/ft_mcap            # 拷出到本地的路径
export S3_ENDPOINT=${S3_ENDPOINT:-http://localhost:9000}   # 宿主机访问 MinIO 映射端口

echo "== [1/6] 造 $N 条 aloha MCAP（/top 480x640 @30Hz + 14 维 joint_states @500Hz）=="
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/*.mcap           # 防旧包重复灌库（episode_id 随机，不去重）
DURS=(6.0 6.5 7.0 7.5 8.0 8.5 9.0 9.5 10.0 10.5)
for i in $(seq 0 $((N - 1))); do
  dur=${DURS[$((i % ${#DURS[@]}))]}
  python3 device/make_sample_bag.py --out-dir "$OUT_DIR" \
    --name "aloha_pick_$(printf %03d $i).mcap" --duration "$dur"
done

echo "== [2/6] 上传 MinIO（bucket 通知 → Kafka → ray-worker 自动解析灌库）=="
python3 device/upload_mcap.py --dir "$OUT_DIR" \
  --embodiment aloha --task pick_red_cube \
  --instruction "pick up the red cube" --source sim --success true

echo "== [3/6] 等待灌库完成（轮询 Iceberg episodes 行数，超时 300s）=="
lake_rows() {
  docker compose exec -T ray-worker python -c "
from robotloop.schema.sink import load_rest_catalog
print(load_rest_catalog().load_table('robotloop.episodes').scan().to_arrow().num_rows)
" 2>/dev/null || echo 0
}
t0=$(date +%s)
while :; do
  rows=$(lake_rows)
  echo "    episodes in lake: $rows / $N"
  [ "$rows" -ge "$N" ] && break
  if [ $(( $(date +%s) - t0 )) -gt 300 ]; then
    echo "❌ 灌库超时（5 分钟）。检查：docker compose logs ray-worker | tail -50"
    exit 1
  fi
  sleep 5
done

echo "== [4/6] 导出 LeRobot v2.1（图像特征 + 归一化统计，fps=30.0 显式声明）=="
docker compose exec -T ray-worker python -c "
from robotloop.retrieval.store import MilvusIcebergStore
from robotloop.export.lerobot_export import export_to_lerobot
store = MilvusIcebergStore()
info = export_to_lerobot(store, '$EXPORT_DIR',
                         filters={'embodiment_tag': 'aloha', 'source': 'sim'}, fps=30.0)
print('exported:', info['total_episodes'], 'episodes /', info['total_frames'], 'frames')
assert info['total_episodes'] >= 1, '导出为空 —— 检查 filters 是否命中'
"

echo "== [5/6] 拷出到本地 $LOCAL_EXPORT =="
rm -rf "$LOCAL_EXPORT"
docker compose cp "ray-worker:$EXPORT_DIR" "$LOCAL_EXPORT"

echo "== [6/6] 训练前校验（模拟 lerobot 加载端全项检查）=="
python3 scripts/verify_lerobot_dataset.py "$LOCAL_EXPORT" --expect-env aloha

cat <<EOF

✅ 数据集就绪并通过校验: $LOCAL_EXPORT

下一步（AutoDL 训练）：
  scp -r $LOCAL_EXPORT root@<AutoDL-IP>:/root/lerobot/data/ft_mcap
  # AutoDL 上（run_act.sh 由 finetune-script 生成，--env.type=aloha 14 维匹配）：
  DATASET=/root/lerobot/data/ft_mcap bash run_act.sh

⚠️ 重跑本脚本前请清理旧 episode（episode_id 随机生成，重灌不去重）：
  # 清 Iceberg 全表 + MinIO frames 目录（仅 demo 环境使用的粗暴清法）：
  docker compose exec -T ray-worker python -c "
from robotloop.schema.sink import load_rest_catalog
t = load_rest_catalog().load_table('robotloop.episodes')
t.delete(delete_filter='fps >= 0')
print('iceberg episodes cleared')"
  docker compose exec -T minio sh -c "rm -rf /data/robotloop-data/frames" || true
EOF
