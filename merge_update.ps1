# merge_update.ps1
# Run this from inside D:\dynamic fix\dynamic-pricing-rl  (on the 'dev' branch)

$source = "D:\final1\dynamic-pricing-rl-dev"
$dest   = Get-Location

if (-Not (Test-Path $source)) {
    Write-Host "ERROR: Source folder not found at $source" -ForegroundColor Red
    Write-Host "Run: dir D:\final1   and fix the `$source path above to match." -ForegroundColor Yellow
    exit 1
}

Write-Host "Copying files from $source into $dest ..." -ForegroundColor Cyan
Copy-Item -Path "$source\*" -Destination $dest -Recurse -Force

Write-Host "`nDone copying. Git status:" -ForegroundColor Cyan
git status

Write-Host "`nStaging and committing..." -ForegroundColor Cyan
git add .
git commit -m "Upgrade to Flask-based AI revenue management platform"

Write-Host "`nAll done. Review the commit with 'git log -1' then push with:" -ForegroundColor Green
Write-Host "  git push origin dev" -ForegroundColor Green