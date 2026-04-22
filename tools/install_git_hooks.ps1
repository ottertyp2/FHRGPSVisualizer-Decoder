param()

$ErrorActionPreference = "Stop"

$repoRoot = git rev-parse --show-toplevel
git config core.hooksPath .githooks

Write-Host "Configured core.hooksPath to .githooks for $repoRoot"
Write-Host "The repository pre-push hook will now check the dual-push origin setup."
