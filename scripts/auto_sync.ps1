# Auto-sync: git add -> commit -> push
# Called by Windows Task Scheduler every 5 hours

$RepoPath = "C:\Users\grant\Quant"
$LogFile = "$RepoPath\scripts\sync.log"

Set-Location $RepoPath

# Check for changes
$status = git status --porcelain 2>&1
if (-not $status) {
    $msg = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | no changes, skipped"
    Add-Content -Path $LogFile -Value $msg
    exit 0
}

# Commit and push
git add -A
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
git commit -m "auto-sync: $timestamp"
git push origin master 2>&1

# Log result
$changedFiles = ($status | Measure-Object -Line).Lines
$msg = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | synced $changedFiles file(s)"
Add-Content -Path $LogFile -Value $msg
