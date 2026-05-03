$ErrorActionPreference='Continue'
$root='C:\Users\WhiteWidow\Documents\GitHub\SlothBrain'
$includeResearch = $false
Set-Location $root

$conns = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($conns) {
  ($conns | Select-Object -ExpandProperty OwningProcess -Unique) | ForEach-Object {
    try { Stop-Process -Id $_ -Force -ErrorAction Stop } catch {}
  }
}

$serverJob = Start-Job -ScriptBlock {
  Set-Location 'C:\Users\WhiteWidow\Documents\GitHub\SlothBrain'
  python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
}

$healthy=$false
for($i=0;$i -lt 75;$i++){
  try { Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -Method Get -TimeoutSec 5 -ErrorAction Stop | Out-Null; $healthy=$true; break }
  catch { Start-Sleep -Milliseconds 1000 }
}
if(-not $healthy){
  'HEALTH_FAIL'
  Receive-Job -Id $serverJob.Id -Keep | Select-Object -Last 120
  Stop-Job -Id $serverJob.Id
  Remove-Job -Id $serverJob.Id
  exit 1
}
'HEALTH_OK'

$tasks = @(
  @{ Name='direct_desktop'; Url='http://127.0.0.1:8000/api/chat/direct'; Body=@{ message='what''s on my desktop?' } },
  @{ Name='direct_web'; Url='http://127.0.0.1:8000/api/chat/direct'; Body=@{ message='Visit https://bytebrew.cc and summarize what the company does in 2 sentences.' } },
  @{ Name='direct_workspace'; Url='http://127.0.0.1:8000/api/chat/direct'; Body=@{ message='Check local workspace: list three files in backend/agents and confirm whether backend/agents/main_agent.py exists.' } },
  @{ Name='chat_status_typo'; Url='http://127.0.0.1:8000/api/chat'; Body=@{ message='/ststus' } },
  @{ Name='chat_status'; Url='http://127.0.0.1:8000/api/chat'; Body=@{ message='/status' } }
)

if ($includeResearch) {
  $tasks += @{ Name='chat_research'; Url='http://127.0.0.1:8000/api/chat'; Body=@{ message='/research Gemma 4 e2b vs qwen 3.6 35b' } }
}

$reqJobs = @()
$jobNames = @{}
foreach($t in $tasks){
  $job = Start-Job -ScriptBlock {
    param($name,$url,$body)
    $result = [ordered]@{ name=$name; ok=$false; status=$null; error=$null; response='' }
    try {
      $json = $body | ConvertTo-Json -Depth 8
      $resp = Invoke-RestMethod -Uri $url -Method Post -Body $json -ContentType 'application/json' -TimeoutSec 120 -ErrorAction Stop
      $result.ok = $true
      $result.status = 200
      $content = ($resp | ConvertTo-Json -Compress)
      if($content.Length -gt 380){ $content = $content.Substring(0,380) }
      $result.response = $content
    } catch {
      $result.error = $_.Exception.Message
      if($_.Exception.Message -match '\((\d{3})\)'){ $result.status = [int]$matches[1] }
    }
    [pscustomobject]$result
  } -ArgumentList $t.Name,$t.Url,$t.Body
  $reqJobs += $job
  $jobNames[$job.Id] = $t.Name
}

Wait-Job -Job $reqJobs -Timeout 130 | Out-Null
$results = @()
foreach($j in $reqJobs){
  if($j.State -eq 'Running'){
    Stop-Job -Id $j.Id | Out-Null
    $taskName = $jobNames[$j.Id]
    $results += [pscustomobject]@{ name=$taskName; ok=$false; status=$null; error='Timed out at 130s'; response='' }
  } else {
    $results += Receive-Job -Id $j.Id
  }
}

'VALIDATION_RESULTS_BEGIN'
$results | Sort-Object name | ForEach-Object {
  "TASK=$($_.name) OK=$($_.ok) STATUS=$($_.status) ERROR=$($_.error)"
  if($_.response){ "RESP=$($_.response)" }
  '---'
}
'VALIDATION_RESULTS_END'

'BACKEND_LOG_TAIL_BEGIN'
Receive-Job -Id $serverJob.Id -Keep | Select-Object -Last 160
'BACKEND_LOG_TAIL_END'

foreach($j in $reqJobs){ try { Remove-Job -Id $j.Id -Force } catch {} }
try { Stop-Job -Id $serverJob.Id } catch {}
try { Remove-Job -Id $serverJob.Id -Force } catch {}

$tailConns = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if($tailConns){
  ($tailConns | Select-Object -ExpandProperty OwningProcess -Unique) | ForEach-Object {
    try { Stop-Process -Id $_ -Force -ErrorAction Stop } catch {}
  }
}
