$researchPayload = @{ message = '/task research sun hydraulics llc and provide sources'; mode = 'auto'; max_steps = 10 } | ConvertTo-Json -Compress
$subagentPayload = @{ message = '/task use a sub agent with preset_id default to summarize what the SlothBrain repository does'; mode = 'auto'; max_steps = 10 } | ConvertTo-Json -Compress

Write-Output 'RESEARCH'
try {
  $research = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/chat' -Method Post -ContentType 'application/json' -Body $researchPayload
  $research | ConvertTo-Json -Compress -Depth 10 | Write-Output
} catch {
  Write-Output $_.Exception.Message
}

Write-Output 'SUBAGENT'
try {
  $subagent = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/chat' -Method Post -ContentType 'application/json' -Body $subagentPayload
  $subagent | ConvertTo-Json -Compress -Depth 10 | Write-Output
} catch {
  Write-Output $_.Exception.Message
}
