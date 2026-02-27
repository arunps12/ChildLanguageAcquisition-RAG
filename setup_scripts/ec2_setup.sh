#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# setup_scripts/ec2_setup.sh — Bootstrap a fresh Ubuntu EC2 instance
# Run once after launching. SSH in, then: bash ec2_setup.sh
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "──── Updating system ────"
sudo apt-get update -y && sudo apt-get upgrade -y

echo "──── Installing Docker ────"
curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
sudo sh /tmp/get-docker.sh
sudo usermod -aG docker "$USER"

echo "──── Installing AWS CLI v2 ────"
if ! command -v aws &>/dev/null; then
  curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  sudo apt-get install -y unzip
  unzip -qo /tmp/awscliv2.zip -d /tmp
  sudo /tmp/aws/install
  rm -rf /tmp/aws /tmp/awscliv2.zip
fi

echo "──── Installing Docker Compose plugin ────"
sudo apt-get install -y docker-compose-plugin 2>/dev/null || true

echo ""
echo "============================================================"
echo " Done! Log out and back in so Docker group takes effect."
echo " Then run:  aws configure"
echo "============================================================"