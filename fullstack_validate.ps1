$ErrorActionPreference='Continue'
$root='C:\Users\WhiteWidow\Documents\GitHub\SlothBrain'
Set-Location $root

$pids = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
foreach($id in $pids){
  try { Stop-Process -Id $id -Force -ErrorAction Stop } catch {}
}

$job = Start-Job -ScriptBlock {
  Set-Location 'C:\Users\WhiteWidow\Documents\GitHub\SlothBrain'
  python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
}

$healthy = $false
for($i=0; $i -lt 90; $i++){
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
  Receive-Job -Id $job.Id -Keep | Select-Object -Last 80
  Stop-Job -Id $job.Id
  Remove-Job -Id $job.Id
  exit 1
}

'HEALTH_OK'

function Call-Api($name, $uri, $payload){
  try {
    $body = $payload | ConvertTo-Json -Depth 8
    $r = Invoke-RestMethod -Uri $uri -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 120 -ErrorAction Stop
    $j = $r | ConvertTo-Json -Compress
    if($j.Length -gt 500){ $j = $j.Substring(0,500) }
    "CASE=$name OK=true RESP=$j"
  } catch {
    "CASE=$name OK=false ERR=$($_.Exception.Message)"
  }
}

try {
  $d0 = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/discord/debug' -Method Get -TimeoutSec 20 -ErrorAction Stop | ConvertTo-Json -Compress
  if($d0.Length -gt 500){ $d0 = $d0.Substring(0,500) }
  "CASE=discord_debug_before OK=true RESP=$d0"
} catch {
  "CASE=discord_debug_before OK=false ERR=$($_.Exception.Message)"
}

Call-Api 'tool_discord_history' 'http://127.0.0.1:8000/api/tools/run' @{ name='discord'; args=@{ action='history'; limit=10 } }
Call-Api 'chat_hi' 'http://127.0.0.1:8000/api/chat' @{ message='hi'; mode='auto'; max_steps=6 }
Call-Api 'chat_url_question' 'http://127.0.0.1:8000/api/chat' @{ message='What''s up with this? https://github.com/ggml-org/llama.cpp'; mode='auto'; max_steps=6 }
Call-Api 'chat_direct_hi' 'http://127.0.0.1:8000/api/chat/direct' @{ message='hi'; mode='direct' }

try {
  $d1 = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/discord/debug' -Method Get -TimeoutSec 20 -ErrorAction Stop | ConvertTo-Json -Compress
  if($d1.Length -gt 500){ $d1 = $d1.Substring(0,500) }
  "CASE=discord_debug_after OK=true RESP=$d1"
} catch {
  "CASE=discord_debug_after OK=false ERR=$($_.Exception.Message)"
}

'RecentBackendLog:'
Receive-Job -Id $job.Id -Keep | Select-Object -Last 160

try { Stop-Job -Id $job.Id } catch {}
try { Remove-Job -Id $job.Id -Force } catch {}
