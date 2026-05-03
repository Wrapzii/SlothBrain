$ErrorActionPreference='Continue'

try {
  $h = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -Method Get -TimeoutSec 5 -ErrorAction Stop
  'HEALTH_OK ' + ($h | ConvertTo-Json -Compress)
} catch {
  'HEALTH_ERR ' + $_.Exception.Message
}

try {
  $d = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/discord/debug' -Method Get -TimeoutSec 10 -ErrorAction Stop
  $j = $d | ConvertTo-Json -Compress
  if ($j.Length -gt 800) { $j = $j.Substring(0,800) }
  'DISCORD_DEBUG ' + $j
} catch {
  'DISCORD_DEBUG_ERR ' + $_.Exception.Message
}

$cases = @(
  @{name='chat_url'; uri='http://127.0.0.1:8000/api/chat'; body=@{ message='What''s up with this? https://github.com/lahfir/agent-desktop'; mode='auto'; max_steps=6 }; timeout=45},
  @{name='chat_status'; uri='http://127.0.0.1:8000/api/chat'; body=@{ message='/status'; mode='auto' }; timeout=20},
  @{name='direct_hi'; uri='http://127.0.0.1:8000/api/chat/direct'; body=@{ message='hi'; mode='direct' }; timeout=20}
)

foreach($c in $cases){
  try {
    $r = Invoke-RestMethod -Uri $c.uri -Method Post -Body ($c.body | ConvertTo-Json -Depth 8) -ContentType 'application/json' -TimeoutSec $c.timeout -ErrorAction Stop
    $j = $r | ConvertTo-Json -Compress
    if ($j.Length -gt 700) { $j = $j.Substring(0,700) }
    "CASE=$($c.name) OK=true RESP=$j"
  } catch {
    "CASE=$($c.name) OK=false ERR=$($_.Exception.Message)"
  }
}
