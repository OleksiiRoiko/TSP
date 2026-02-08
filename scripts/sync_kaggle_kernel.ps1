$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

Write-Host "Syncing Kaggle kernel bundle..." -ForegroundColor Cyan

robocopy src kaggle\kernel\src /E /XD __pycache__ | Out-Null
robocopy configs kaggle\kernel\configs /E | Out-Null
Copy-Item -Path requirements.txt -Destination kaggle\kernel\requirements.txt -Force
Copy-Item -Path config.yaml -Destination kaggle\kernel\config.yaml -Force

# Build fallback bundle for Kaggle (single zip)
$bundle = Join-Path $repoRoot "kaggle\kernel\project.zip"
if (Test-Path $bundle) { Remove-Item $bundle -Force }
Compress-Archive -Path src,configs,requirements.txt,config.yaml,("kaggle\kernel\configs_to_run.txt") -DestinationPath $bundle -Force

Write-Host "Done. You can now run: kaggle kernels push -p kaggle\kernel" -ForegroundColor Green
