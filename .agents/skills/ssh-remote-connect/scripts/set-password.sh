#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SSH_REMOTE_CONFIG:-$SCRIPT_DIR/connection.local.env}"
TEMPLATE_FILE="$SCRIPT_DIR/connection.env.example"

if [[ ! -f "$CONFIG_FILE" ]]; then
  cp "$TEMPLATE_FILE" "$CONFIG_FILE"
fi

printf "SSH password: " >&2
IFS= read -r -s PASSWORD
printf "\n" >&2

python3 - "$CONFIG_FILE" "$PASSWORD" <<'PY'
import pathlib
import shlex
import sys

path = pathlib.Path(sys.argv[1])
password = sys.argv[2]

lines = path.read_text().splitlines()
updated = []
seen = False

for line in lines:
    if line.startswith("SSH_PASSWORD="):
        updated.append(f"SSH_PASSWORD={shlex.quote(password)}")
        seen = True
    else:
        updated.append(line)

if not seen:
    updated.append(f"SSH_PASSWORD={shlex.quote(password)}")

path.write_text("\n".join(updated) + "\n")
path.chmod(0o600)
PY

echo "Password updated in $CONFIG_FILE" >&2
