#!/bin/sh
echo "========== Generating robot data =========="
python generator.py
echo "========== Uploading data to MinIO =========="
python uploader.py
echo "========== Done =========="