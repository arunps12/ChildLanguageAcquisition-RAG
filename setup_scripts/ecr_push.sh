#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# setup_scripts/ecr_push.sh — Build Docker image and push to Amazon ECR
#
# Usage:
#   bash setup_scripts/ecr_push.sh              # uses defaults
#   AWS_REGION=eu-north-1 ECR_REPO=childlanguagenet-rag bash setup_scripts/ecr_push.sh
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

AWS_REGION="${AWS_REGION:-eu-north-1}"
ECR_REPO="${ECR_REPO:-childlanguagenet-rag}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

echo "──── ECR Push ────"
echo "  Region:   ${AWS_REGION}"
echo "  Repo:     ${ECR_REPO}"
echo "  URI:      ${ECR_URI}:${IMAGE_TAG}"
echo ""

# 1. Create ECR repo if it doesn't exist
aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${AWS_REGION}" 2>/dev/null \
  || aws ecr create-repository \
       --repository-name "${ECR_REPO}" \
       --region "${AWS_REGION}" \
       --image-scanning-configuration scanOnPush=true

# 2. Authenticate Docker to ECR
echo "Authenticating Docker to ECR …"
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# 3. Build the Docker image
echo "Building Docker image …"
docker build -t "${ECR_REPO}:${IMAGE_TAG}" .

# 4. Tag for ECR
docker tag "${ECR_REPO}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"

# 5. Push to ECR
echo "Pushing to ECR …"
docker push "${ECR_URI}:${IMAGE_TAG}"

echo ""
echo "============================================================"
echo " Pushed: ${ECR_URI}:${IMAGE_TAG}"
echo "============================================================"
