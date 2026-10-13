#!/usr/bin/env bash
set -euo pipefail
deploy_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "${EMBOSCOPE_PYTHON:-python3}" "$deploy_root/install.py" "$@"
