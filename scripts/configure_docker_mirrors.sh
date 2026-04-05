#!/bin/bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Please run as root: sudo bash $0" >&2
  exit 1
fi

mkdir -p /etc/docker

cat >/etc/docker/daemon.json <<EOF
{
  "registry-mirrors": [
    "https://docker.1ms.run",
    "https://hub.rat.dev",
    "https://dockerpull.com"
  ]
}
EOF

systemctl daemon-reload
systemctl restart docker

echo
echo "Docker registry mirrors configured."
echo "Verify with:"
echo "  docker info | grep -A5 'Registry Mirrors'"
echo "  docker run hello-world"
