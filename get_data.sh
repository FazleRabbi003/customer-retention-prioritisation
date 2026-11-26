#!/usr/bin/env bash
# Fetches the IBM Telco Customer Churn dataset (public).
set -euo pipefail
mkdir -p churn_data
curl -sSL -o churn_data/telco.csv \
  "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
echo "downloaded churn_data/telco.csv ($(wc -l < churn_data/telco.csv) lines)"
