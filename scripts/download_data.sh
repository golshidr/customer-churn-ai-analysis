#!/usr/bin/env bash
# Download the IBM Telco Customer Churn data to data/raw/telco_customer_churn.csv.
# Tries Kaggle (via kagglehub) first and falls back to IBM's public GitHub copy;
# the file is verified against a SHA-256 checksum either way.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m churn_analysis download "$@"
