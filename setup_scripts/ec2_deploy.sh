#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# setup_scripts/ec2_deploy.sh — Pull image from ECR and run on EC2
#
# Run this ON the EC2 instance after ec2_setup.sh and aws configure.
#
# Usage:
#   bash ec2_deploy.sh
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

AWS_REGION="${AWS_REGION:-eu-north-1}"
ECR_REPO="${ECR_REPO:-childlanguagenet-rag}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_NAME="childlanguagenet"

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

echo "──── EC2 Deploy ────"
echo "  Image:  ${ECR_URI}:${IMAGE_TAG}"
echo ""

# 1. Authenticate Docker to ECR
echo "Authenticating Docker to ECR …"
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# 2. Pull the latest image
echo "Pulling image …"
docker pull "${ECR_URI}:${IMAGE_TAG}"

# 3. Stop old container if running
echo "Stopping old container (if any) …"
docker stop "${CONTAINER_NAME}" 2>/dev/null || true
docker rm "${CONTAINER_NAME}" 2>/dev/null || true

# 4. Check .env file exists
if [[ ! -f .env ]]; then
  echo "ERROR: .env file not found in $(pwd)"
  echo "Create one with: OPENAI_API_KEY=sk-..."
  exit 1
fi

# 5. Run the new container
echo "Starting container …"
docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  -p 8501:8501 \
  --env-file .env \
  "${ECR_URI}:${IMAGE_TAG}"

echo ""
echo "============================================================"
echo " App is running at http://<your-ec2-public-ip>:8501"
echo " Container: ${CONTAINER_NAME}"
echo " Logs:      docker logs -f ${CONTAINER_NAME}"
echo "============================================================"
