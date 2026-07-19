#!/bin/sh
echo "========== Generating sample MCAP bag =========="
python /app/make_sample_bag.py --out-dir /app/sample_bags

echo "========== Uploading sample bag to MinIO =========="
python /app/upload_mcap.py --dir /app/sample_bags \
    --embodiment aloha --task pick_red_cube \
    --instruction "pick up the red cube" --source sim --success true

echo "========== Generating robot data =========="
python generator.py
echo "========== Uploading data to MinIO =========="
python uploader.py
echo "========== Done =========="

# 保持容器运行，便于后续 exec 手动上传真实采集的包
tail -f /dev/null