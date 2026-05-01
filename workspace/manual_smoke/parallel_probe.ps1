$payload1 = @{ message = '/task what is my user name'; mode = 'auto'; max_steps = 10 } | ConvertTo-Json -Compress
$payload2 = @{ message = '/task what is my computers name'; mode = 'auto'; max_steps = 10 } | ConvertTo-Json -Compress
$j1=Start-Job -ScriptBlock {
	param($body)
	Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/chat' -Method Post -ContentType 'application/json' -Body $body | ConvertTo-Json -Compress
} -ArgumentList $payload1
$j2=Start-Job -ScriptBlock {
	param($body)
	Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/chat' -Method Post -ContentType 'application/json' -Body $body | ConvertTo-Json -Compress
} -ArgumentList $payload2
Wait-Job -Job $j1,$j2 -Timeout 60 | Out-Null
$o1=Receive-Job -Job $j1
$o2=Receive-Job -Job $j2
Write-Output "R1"
Write-Output $o1
Write-Output "R2"
Write-Output $o2
