#!/usr/bin/env bash
set -euo pipefail

ROLES=(
  "Software Engineer"
  "Data Scientist"
  "ML Engineer"
  "DevOps Engineer"
  "Product Manager"
  "Financial Analyst"
  "HR Specialist"
  "Retail Associate"
  "Sales Representative"
  "Customer Support"
)

OUT_BASE="/home/zs7353/resume_validity/data/harvest_top10"
COMPANIES="stripe,datadog,asana,instacart,roblox,opendoor,brex"
MAX_JOBS=200
TARGET=100

mkdir -p "$OUT_BASE"

for role in "${ROLES[@]}"; do
  slug=$(echo "$role" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
  sess="harvest_${slug}"
  out_dir="$OUT_BASE/$slug"
  mkdir -p "$out_dir"
  tmux new-session -d -s "$sess" -- bash -lc "export GOOGLE_API_KEY=\"${GOOGLE_API_KEY:-$GOOGLE_API_KEY}\"; python3 -m resume_validity.scrape.harvest_pass_postings --companies $COMPANIES --roles \"$role\" --max_jobs_per_company $MAX_JOBS --per_role_target $TARGET --require_basic 0 --require_bonus 0 --out_dir $out_dir --no_progress"
  echo "Launched $sess -> $out_dir"
done

tmux ls | grep harvest_ || true


