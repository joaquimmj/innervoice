#!/usr/bin/env bash
# Block commits that contain token-like strings. Usage: check_secrets.sh [files...] (default: staged files)
set -euo pipefail
PATTERN='(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|gho_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|"key"[[:space:]]*:[[:space:]]*"[a-f0-9]{32}"|KAGGLE_KEY[[:space:]]*=[[:space:]]*[A-Za-z0-9]{8,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)'
if [ "$#" -gt 0 ]; then files=("$@"); else mapfile -t files < <(git diff --cached --name-only --diff-filter=ACM); fi
hits=0
for f in "${files[@]}"; do
  [ -f "$f" ] || continue
  if grep -IEnq "$PATTERN" "$f"; then echo "check_secrets: token-like string in $f" >&2; hits=1; fi
done
exit $hits
