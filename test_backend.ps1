try {
    $process = Start-Process uvicorn -ArgumentList "backend.main:app", "--host", "127.0.0.1", "--port", "8000" -PassThru -NoNewWindow
    Write-Host "Started uvicorn with PID: $($process.Id)"
    
    $start = Get-Date
    $success = $false
    while (((Get-Date) - $start).TotalSeconds -lt 45) {
        try {
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get -ErrorAction Stop
            Write-Host "Health check success"
            $success = $true
            break
        } catch {
            Start-Sleep -Seconds 2
        }
    }
    
    if (-not $success) {
        Write-Error "Backend failed to start within 45s"
        exit 1
    }
    
    $messages = @(
        "what's on my desktop?",
        "Visit https://bytebrew.cc and summarize what the company does in 2 sentences.",
        "Check local workspace: list three files in backend/agents and confirm whether backend/agents/main_agent.py exists."
    )
    
    foreach ($msg in $messages) {
        Write-Host "Sending message: $msg"
        try {
            $body = @{ message = $msg } | ConvertTo-Json
            $response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/chat/direct" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 30
            $status = $response.StatusCode
            $content = $response.Content
            if ($content.Length -gt 260) { $content = $content.Substring(0, 260) }
            Write-Host "Status: $status"
            Write-Host "Response: $content"
        } catch {
            Write-Host "Request failed: $_"
        }
        Write-Host "-------------------"
    }
} finally {
    if ($process) {
        Write-Host "Terminating uvicorn process..."
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    }
}
