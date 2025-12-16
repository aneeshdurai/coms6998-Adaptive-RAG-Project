# PowerShell script to stop Streamlit server
Write-Host "Stopping Streamlit server..." -ForegroundColor Yellow

# Kill processes by name
Get-Process | Where-Object {$_.ProcessName -like "*python*" -or $_.ProcessName -like "*streamlit*"} | 
    Where-Object {$_.Path -like "*dpo_workspace*" -or $_.CommandLine -like "*streamlit*"} | 
    Stop-Process -Force -ErrorAction SilentlyContinue

# Kill process using port 8501
$port = 8501
$connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
foreach ($conn in $connections) {
    if ($conn.State -eq "Listen") {
        $pid = $conn.OwningProcess
        Write-Host "Killing process on port $port (PID: $pid)" -ForegroundColor Yellow
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "Done. If server is still running, check Task Manager." -ForegroundColor Green

