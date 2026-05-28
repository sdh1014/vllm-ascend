#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SSH_REMOTE_CONFIG:-$SCRIPT_DIR/connection.local.env}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Missing SSH config: $CONFIG_FILE" >&2
  echo "Create it from: $SCRIPT_DIR/connection.env.example" >&2
  exit 1
fi

set -a
# shellcheck source=/dev/null
. "$CONFIG_FILE"
set +a

: "${SSH_HOST:?Missing SSH_HOST in $CONFIG_FILE}"
: "${SSH_PORT:?Missing SSH_PORT in $CONFIG_FILE}"
: "${SSH_USER:?Missing SSH_USER in $CONFIG_FILE}"

SSH_TARGET="${SSH_USER}@${SSH_HOST}"
SSH_ARGS=(-p "$SSH_PORT" "$SSH_TARGET")

if [[ -n "${SSH_PASSWORD:-}" ]] && command -v sshpass >/dev/null 2>&1; then
  export SSHPASS="$SSH_PASSWORD"
  exec sshpass -e ssh "${SSH_ARGS[@]}" "$@"
fi

exec ssh "${SSH_ARGS[@]}" "$@"
