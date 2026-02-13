# 自动同步脚本：git add → commit → push
# 由 Windows 任务计划程序每5小时调用一次

$RepoPath = "C:\Users\grant\Quant"
$LogFile = "$RepoPath\scripts\sync.log"

Set-Location $RepoPath

# 检查是否有变更
$status = git status --porcelain 2>&1
if (-not $status) {
    $msg = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | 无变更，跳过"
    Add-Content -Path $LogFile -Value $msg
    exit 0
}

# 提交并推送
git add -A
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
git commit -m "auto-sync: $timestamp"
git push origin master 2>&1

# 记录日志
$changedFiles = ($status | Measure-Object -Line).Lines
$msg = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | 已同步 $changedFiles 个文件变更"
Add-Content -Path $LogFile -Value $msg
