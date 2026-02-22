#!/usr/bin/env bash
set -euo pipefail

IMAGE="gcr.io/pawfect-edit/pawfect-edit-inpaint"
REGION="us-east4"
SERVICE="pawfect-edit-inpaint"

# Usage: ./scripts/deploy.sh [--tiny]
TINY_MODEL=0
for arg in "$@"; do
  case $arg in
    --tiny) TINY_MODEL=1 ;;
  esac
done

# Resolve HF_TOKEN: .env.local > environment > ~/.cache/huggingface/token
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env.local"
if [ -f "$ENV_FILE" ]; then
  HF_TOKEN="${HF_TOKEN:-$(grep '^HF_TOKEN=' "$ENV_FILE" | cut -d= -f2-)}"
fi
HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null || true)}"
if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN not found in .env.local, environment, or ~/.cache/huggingface/token"
  exit 1
fi

if [ "$TINY_MODEL" = "1" ]; then
  echo "==> TINY MODEL mode (fast build, garbage output)"
fi

echo "==> Building linux/amd64 image..."
docker buildx build --platform linux/amd64 \
  --build-arg HF_TOKEN="$HF_TOKEN" \
  --build-arg TINY_MODEL="$TINY_MODEL" \
  -t "$IMAGE" \
  --load \
  .

echo "==> Pushing to GCR..."
docker push "$IMAGE"

echo "==> Deploying to Cloud Run..."
gcloud run deploy "$SERVICE" \
  --image="$IMAGE" \
  --region="$REGION" \
  --platform=managed \
  --no-allow-unauthenticated \
  --gpu=1 --gpu-type=nvidia-l4 --no-gpu-zonal-redundancy \
  --cpu=8 --memory=32Gi \
  --max-instances=1 --min-instances=0 --concurrency=1 \
  --timeout=3600 \
  --set-env-vars=TINY_MODEL="$TINY_MODEL" \
  --set-secrets=FIREBASE_CREDENTIALS=FIREBASE_CREDENTIALS:latest,HF_TOKEN=HF_TOKEN:latest \
  --startup-probe=httpGet.path=/ready,initialDelaySeconds=10,periodSeconds=10,failureThreshold=60,timeoutSeconds=5 \
  --liveness-probe=httpGet.path=/health

echo "==> Done. Service URL:"
gcloud run services describe "$SERVICE" --region="$REGION" --format="value(status.url)"
