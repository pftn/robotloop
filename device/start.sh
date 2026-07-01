#!/bin/sh
echo "========== 开始生成数据 =========="
python generator.py
echo "生成脚本退出码: $?"

echo "========== 开始上传数据 =========="
python uploader.py
echo "上传脚本退出码: $?"

echo "========== 全部完成 =========="
tail -f /dev/null