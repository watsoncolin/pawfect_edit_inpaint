#!/usr/bin/env bash
set -euo pipefail

IMAGE="gcr.io/pawfect-edit/pawfect-edit-inpaint"
REGION="us-east4"
SERVICE="pawfect-edit-inpaint"

# Resolve HF_TOKEN
HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null || true)}"
if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN not set and ~/.cache/huggingface/token not found"
  exit 1
fi

echo "==> Building linux/amd64 image..."
docker buildx build --platform linux/amd64 \
  --build-arg HF_TOKEN="$HF_TOKEN" \
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
  --set-secrets=FIREBASE_CREDENTIALS=FIREBASE_CREDENTIALS:latest,HF_TOKEN=HF_TOKEN:latest \
  --startup-probe=httpGet.path=/ready,initialDelaySeconds=10,periodSeconds=10,failureThreshold=60,timeoutSeconds=5 \
  --liveness-probe=httpGet.path=/health

echo "==> Done. Service URL:"
gcloud run services describe "$SERVICE" --region="$REGION" --format="value(status.url)"
