#!/bin/bash
set -e

# ============================================
# Ray Worker Node Entrypoint
# ============================================
# 1. Waits for head node to be ready
# 2. Joins worker to the Ray cluster
# 3. Runs the pipeline
# 4. Gracefully exits on termination

HEAD_HOST="${RAY_HEAD_HOST:-ray-head}"
HEAD_PORT="${RAY_HEAD_PORT:-6379}"
HEAD_ADDRESS="${HEAD_HOST}:${HEAD_PORT}"
MAX_RETRIES=30
RETRY_DELAY=2

echo "=========================================="
echo "🚀 Ray Worker Node Starting"
echo "   Head: ${HEAD_ADDRESS}"
echo "=========================================="

# --- Step 1: Wait for head node ---
echo "⏳ Waiting for Ray head node at ${HEAD_ADDRESS}..."
retry_count=0
while ! nc -z "${HEAD_HOST}" "${HEAD_PORT}" 2>/dev/null; do
    retry_count=$((retry_count + 1))
    if [ $retry_count -ge $MAX_RETRIES ]; then
        echo "❌ Failed to connect to head node after ${MAX_RETRIES} attempts"
        exit 1
    fi
    echo "   Retry ${retry_count}/${MAX_RETRIES}..."
    sleep $RETRY_DELAY
done
echo "✅ Head node is reachable"

# --- Step 2: Extra wait for GCS to fully initialize ---
sleep 3

# --- Step 3: Join the Ray cluster ---
echo "🔗 Joining Ray cluster at ${HEAD_ADDRESS}..."
ray start \
    --address="${HEAD_ADDRESS}" \
    --num-cpus="${RAY_NUM_CPUS:-1}" \
    --block &

RAY_PID=$!

sleep 5

# Verify connection
echo "🔍 Verifying cluster connection..."
if ! ray status >/dev/null 2>&1; then
    echo "❌ Failed to join cluster"
    kill $RAY_PID 2>/dev/null || true
    exit 1
fi

echo "✅ Successfully joined Ray cluster"
ray status

# --- Step 4: Run the pipeline ---
echo ""
echo "=========================================="
echo "🏃 Starting pipeline: ray_pipeline.py"
echo "=========================================="

python /app/ray_pipeline.py

PIPELINE_EXIT_CODE=$?

# --- Step 5: Cleanup ---
echo ""
echo "🛑 Stopping Ray worker..."
ray stop

if [ $PIPELINE_EXIT_CODE -ne 0 ]; then
    echo "❌ Pipeline exited with code ${PIPELINE_EXIT_CODE}"
    exit $PIPELINE_EXIT_CODE
fi

echo "✅ Pipeline completed successfully"