#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# setup_scripts/jenkins.sh — Install Jenkins on a dedicated EC2 instance
# Run once: bash jenkins.sh
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "──── Updating system ────"
sudo apt-get update -y && sudo apt-get upgrade -y

# ── Java (Jenkins requirement) ──
echo "──── Installing Java 17 ────"
sudo apt-get install -y fontconfig openjdk-17-jre

# ── Jenkins ──
echo "──── Installing Jenkins ────"
sudo wget -O /usr/share/keyrings/jenkins-keyring.asc \
  https://pkg.jenkins.io/debian-stable/jenkins.io-2023.key
echo "deb [signed-by=/usr/share/keyrings/jenkins-keyring.asc] \
  https://pkg.jenkins.io/debian-stable binary/" \
  | sudo tee /etc/apt/sources.list.d/jenkins.list > /dev/null
sudo apt-get update -y
sudo apt-get install -y jenkins

sudo systemctl enable jenkins
sudo systemctl start jenkins

# ── Docker ──
echo "──── Installing Docker ────"
curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
sudo sh /tmp/get-docker.sh
sudo usermod -aG docker "$USER"
sudo usermod -aG docker jenkins

# ── AWS CLI v2 ──
echo "──── Installing AWS CLI v2 ────"
if ! command -v aws &>/dev/null; then
  curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  sudo apt-get install -y unzip
  unzip -qo /tmp/awscliv2.zip -d /tmp
  sudo /tmp/aws/install
  rm -rf /tmp/aws /tmp/awscliv2.zip
fi

# ── Restart Jenkins to pick up Docker group ──
sudo systemctl restart jenkins

echo ""
echo "============================================================"
echo " Jenkins installed!"
echo ""
echo " 1. Get admin password:"
echo "    sudo cat /var/lib/jenkins/secrets/initialAdminPassword"
echo ""
echo " 2. Open Jenkins at:"
echo "    http://<this-server-ip>:8080"
echo ""
echo " 3. Run: aws configure"
echo "    (for Jenkins user: sudo -u jenkins aws configure)"
echo ""
echo " 4. Add these Jenkins credentials (Manage Jenkins → Credentials):"
echo "      - AWS_REGION            (Secret text, e.g. eu-north-1)"
echo "      - AWS_ACCOUNT_ID        (Secret text, e.g. 123456789012)"
echo "      - ECR_REPOSITORY        (Secret text, e.g. childlanguagenet-rag)"
echo "      - AWS_ACCESS_KEY_ID     (Secret text)"
echo "      - AWS_SECRET_ACCESS_KEY (Secret text)"
echo "      - APP_HOST              (Secret text, App EC2 public IP)"
echo "      - OPENAI_API_KEY        (Secret text)"
echo "      - ssh_key               (SSH private key for App EC2)"
echo "============================================================"