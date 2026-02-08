$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

Write-Host "Syncing Kaggle kernel bundle..." -ForegroundColor Cyan

# Keep kernel folder lean: Kaggle runtime receives `script.py`, and project code
# is delivered via `project.zip` from dataset source.
if (Test-Path "kaggle\kernel\src") { Remove-Item "kaggle\kernel\src" -Recurse -Force }
if (Test-Path "kaggle\kernel\configs") { Remove-Item "kaggle\kernel\configs" -Recurse -Force }

Copy-Item -Path requirements.txt -Destination kaggle\kernel\requirements.txt -Force
Copy-Item -Path config.yaml -Destination kaggle\kernel\config.yaml -Force

# Build fallback bundle for Kaggle (single zip)
$bundle = Join-Path $repoRoot "kaggle\kernel\project.zip"
if (Test-Path $bundle) { Remove-Item $bundle -Force }
Compress-Archive -Path src,configs,requirements.txt,config.yaml,("kaggle\kernel\configs_to_run.txt") -DestinationPath $bundle -Force

# Keep a copy for Kaggle code dataset.
$codeDatasetDir = Join-Path $repoRoot "kaggle\code_dataset"
if (!(Test-Path $codeDatasetDir)) { New-Item -ItemType Directory -Path $codeDatasetDir | Out-Null }
Copy-Item -Path $bundle -Destination (Join-Path $codeDatasetDir "project.zip") -Force

Write-Host "Done. Next commands:" -ForegroundColor Green
Write-Host "  kaggle datasets version -p kaggle\data_dataset --dir-mode zip -m `"data update`"" -ForegroundColor Green
Write-Host "  kaggle datasets version -p kaggle\code_dataset -m `"code update`"" -ForegroundColor Green
Write-Host "  kaggle kernels push -p kaggle\kernel" -ForegroundColor Green
