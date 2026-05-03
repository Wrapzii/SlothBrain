$ErrorActionPreference='Continue'

$pids = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
foreach($id in $pids){
  try { Stop-Process -Id $id -Force -ErrorAction Stop } catch {}
}

$job = Start-Job -ScriptBlock {
  Set-Location 'C:\Users\WhiteWidow\Documents\GitHub\SlothBrain'
  python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
}

$healthy = $false
for($i=0; $i -lt 60; $i++){
  try {
    Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -Method Get -TimeoutSec 3 -ErrorAction Stop | Out-Null
    $healthy = $true
    break
  } catch {
    Start-Sleep -Seconds 1
  }
}

if(-not $healthy){
  'HEALTH_FAIL'
  Receive-Job -Id $job.Id -Keep | Select-Object -Last 60
  Stop-Job -Id $job.Id
  Remove-Job -Id $job.Id
  exit 1
}

$start = Get-Date
try {
  $resp = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/chat/direct' -Method Post -Body (@{ message='hi'; mode='direct' } | ConvertTo-Json) -ContentType 'application/json' -TimeoutSec 20 -ErrorAction Stop
  $elapsed = [int]((Get-Date) - $start).TotalMilliseconds
  "OK_MS=$elapsed"
  ($resp | ConvertTo-Json -Compress)
} catch {
  $elapsed = [int]((Get-Date) - $start).TotalMilliseconds
  "ERR_MS=$elapsed MSG=$($_.Exception.Message)"
}

'RecentBackendLog:'
Receive-Job -Id $job.Id -Keep | Select-Object -Last 100

try { Stop-Job -Id $job.Id } catch {}
try { Remove-Job -Id $job.Id -Force } catch {}
