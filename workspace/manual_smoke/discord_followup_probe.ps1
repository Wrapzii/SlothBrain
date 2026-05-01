$ErrorActionPreference = 'Stop'

function Invoke-Tool($name, $toolArgs) {
  $body = @{ name = $name; args = $toolArgs } | ConvertTo-Json -Depth 8 -Compress
  return Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/tools/run' -Method Post -ContentType 'application/json' -Body $body
}

function Get-History {
  $r = Invoke-Tool 'discord' @{ action = 'history'; limit = 20 }
  if (-not $r.ok) {
    throw "history failed: $($r.error)"
  }
  return $r.output.messages
}

function Last-UserId {
  $msgs = Get-History
  $u = $msgs | Where-Object { -not $_.is_bot } | Select-Object -First 1
  if ($null -eq $u) { return '' }
  return [string]$u.id
}

function Wait-ForBotAfter([string]$userMsgId, [int]$timeoutSec) {
  $deadline = (Get-Date).AddSeconds($timeoutSec)
  while ((Get-Date) -lt $deadline) {
    $msgs = Get-History
    foreach ($m in $msgs) {
      if ($m.is_bot -and $m.content -and ([string]$m.id -gt $userMsgId)) {
        return $m
      }
    }
    Start-Sleep -Milliseconds 800
  }
  return $null
}

$tests = @(
  'live-check-1 hello from probe',
  'live-check-2 follow-up can you respond once?',
  'live-check-3 second follow-up still there?'
)

$results = @()
foreach ($msg in $tests) {
  $beforeUser = Last-UserId
  $send = Invoke-Tool 'discord' @{ action = 'send'; content = $msg }
  if (-not $send.ok) {
    throw "send failed: $($send.error)"
  }

  $reply = Wait-ForBotAfter $beforeUser 35
  $results += [pscustomobject]@{
    sent = $msg
    got_reply = [bool]$reply
    reply_preview = if ($reply) { ([string]$reply.content -replace "`r|`n", ' ') } else { '' }
    reply_id = if ($reply) { [string]$reply.id } else { '' }
    reply_author = if ($reply) { [string]$reply.author } else { '' }
  }
}

$results | ConvertTo-Json -Depth 6
