#!/usr/bin/env bash
# Provision Grafana Cloud: data source + dashboard (named USERNAME_DICODING) + alert rules.
# Usage:
#   set -a && source .env && set +a
#   bash "Monitoring dan Logging/setup_grafana.sh"
set -euo pipefail

: "${GRAFANA_URL:?set GRAFANA_URL=https://<stack-slug>.grafana.net}"
: "${GRAFANA_TOKEN:?set GRAFANA_TOKEN=glc_...}"
: "${USERNAME_DICODING:?set USERNAME_DICODING=APC005D6Y0216}"
: "${PROMETHEUS_URL:=http://localhost:9090}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DASHBOARD_JSON="$ROOT/Monitoring dan Logging/dashboard.json"
ALERTS_JSON="$ROOT/Monitoring dan Logging/grafana_alerts.json"

API="$GRAFANA_URL/api"
H_AUTH="Authorization: Bearer $GRAFANA_TOKEN"
H_CT="Content-Type: application/json"

echo "==[1/4]== Verify Grafana reachable"
curl -sS -H "$H_AUTH" "$API/org" | head -1

echo "==[2/4]== Create Prometheus datasource"
curl -sS -X POST -H "$H_AUTH" -H "$H_CT" \
    "$API/datasources" \
    -d "{\"name\":\"Prometheus\",\"type\":\"prometheus\",\"url\":\"$PROMETHEUS_URL\",\"access\":\"proxy\",\"isDefault\":true}"
echo

echo "==[3/4]== Import dashboard named $USERNAME_DICODING"
python -c "
import json, os, sys
p='$DASHBOARD_JSON'
j=json.load(open(p))
j['dashboard']['title']=os.environ['USERNAME_DICODING']
print(json.dumps(j))
" > "$ROOT/.dashboard_payload.json"
curl -sS -X POST -H "$H_AUTH" -H "$H_CT" \
    "$API/dashboards/db" \
    --data-binary "@$ROOT/.dashboard_payload.json" | head -200
echo
rm -f "$ROOT/.dashboard_payload.json"

echo "==[4/4]== Create alert rule group ($USERNAME_DICODING)"
python -c "
import json, os, sys
p='$ALERTS_JSON'
j=json.load(open(p))
folder='$USERNAME_DICODING'
print(json.dumps({'folder':folder,'name':j['name'],'interval':'1m','rules':j['rules']}))
" > "$ROOT/.alerts_payload.json"
curl -sS -X POST -H "$H_AUTH" -H "$H_CT" \
    "$API/v1/provisioning/alert-rules" \
    --data-binary "@$ROOT/.alerts_payload.json" | head -100
rm -f "$ROOT/.alerts_payload.json"

echo "Selesai."
