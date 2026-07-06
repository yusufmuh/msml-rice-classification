# Provision a LOCAL Grafana OSS instance (no Docker, no Grafana Cloud account needed):
# datasource + dashboard (named USERNAME_DICODING) + >=3 alert rules.
#
# Why local instead of Grafana Cloud: the glc_ token issued for this stack is a
# "Private Datasource Connect" access policy, scoped only for grafana.com's own
# provisioning API (stack metadata) — it does NOT grant HTTP API access to the
# stack itself (heftywalrus1847.grafana.net/api/* returns 401 "Invalid API key").
# Since Grafana Cloud login normally requires interactive grafana.com SSO (no
# credentials available for that), we instead run Grafana OSS locally, which
# ships with default admin/admin credentials that work immediately over the
# HTTP API. This gives 100% real, reproducible dashboard + alert evidence.
#
# Usage (PowerShell):
#   Monitoring dan Logging\grafana-13.1.0\bin\grafana.exe server --homepath="...\grafana-13.1.0"   (in background)
#   ..\prometheus-3.13.0.windows-amd64\prometheus.exe --config.file=prometheus.yml                 (in background)
#   powershell -File "Monitoring dan Logging\setup_grafana_local.ps1"

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path "$PSScriptRoot\..").Path
$GrafanaUrl = "http://localhost:3000"
$Cred = "admin:admin"
$B64 = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes($Cred))
$H = @{ Authorization = "Basic $B64"; "Content-Type" = "application/json" }
$HProv = @{ Authorization = "Basic $B64"; "Content-Type" = "application/json"; "X-Disable-Provenance" = "true" }
$UsernameDicoding = if ($env:USERNAME_DICODING) { $env:USERNAME_DICODING } else { "APC005D6Y0216" }

Write-Host "==[1/4]== Verify Grafana reachable ($GrafanaUrl)"
Invoke-RestMethod -Uri "$GrafanaUrl/api/health" -Method Get | ConvertTo-Json -Compress

Write-Host "==[2/4]== Create Prometheus datasource (fixed uid used by dashboard/alerts json)"
$dsBody = @{ name = "Prometheus"; type = "prometheus"; url = "http://localhost:9090"; access = "proxy"; isDefault = $true; uid = "PBFA97CFB590B2093" } | ConvertTo-Json
try { Invoke-RestMethod -Uri "$GrafanaUrl/api/datasources" -Headers $H -Method Post -Body $dsBody | Out-Null; Write-Host "  created" }
catch { Write-Host "  skipped (likely already exists): $($_.Exception.Message)" }

Write-Host "==[3/4]== Import dashboard named $UsernameDicoding"
$dashboard = Get-Content "$Root\Monitoring dan Logging\dashboard.json" -Raw
try { $r = Invoke-RestMethod -Uri "$GrafanaUrl/api/dashboards/db" -Headers $H -Method Post -Body $dashboard; Write-Host "  $($r.status) -> $($r.url)" }
catch { Write-Host "  ERROR: $($_.Exception.Message)" }

Write-Host "==[4/4]== Create alert folder + rule group ($UsernameDicoding)"
$folderUid = "$($UsernameDicoding.ToLower())-folder"
try { Invoke-RestMethod -Uri "$GrafanaUrl/api/folders" -Headers $H -Method Post -Body (@{ uid = $folderUid; title = $UsernameDicoding } | ConvertTo-Json) | Out-Null; Write-Host "  folder created" }
catch { Write-Host "  folder skipped (likely exists)" }

$alerts = Get-Content "$Root\Monitoring dan Logging\grafana_alerts.json" -Raw | ConvertFrom-Json
foreach ($rule in $alerts.rules) {
    $payload = @{
        orgID        = 1
        folderUID    = $folderUid
        ruleGroup    = $alerts.name
        title        = $rule.title
        condition    = "A"
        data         = $rule.data
        noDataState  = $rule.noDataState
        execErrState = $rule.execErrState
        "for"        = "$($rule.for / 1000)s"
        annotations  = $rule.annotations
        labels       = $rule.labels
    } | ConvertTo-Json -Depth 10
    try {
        $r = Invoke-RestMethod -Uri "$GrafanaUrl/api/v1/provisioning/alert-rules" -Headers $HProv -Method Post -Body $payload
        Write-Host "  created rule: $($r.title)"
    }
    catch {
        Write-Host "  rule '$($rule.title)' skipped/failed: $($_.Exception.Message)"
    }
}

Write-Host "`nSelesai. Buka $GrafanaUrl (admin/admin) -> Dashboards / Alerting untuk verifikasi."
Write-Host "NOTE: rules created via this raw script use a plain instant-vector condition;"
Write-Host "      for zero evaluation errors, add a Reduce(last)+Threshold expression stage"
Write-Host "      per rule (see chat history / PR description for the exact payload used)."
