param(
  [switch]$SkipDataZip,
  [switch]$ForceDataZip
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

Write-Host "Syncing Kaggle kernel bundle..." -ForegroundColor Cyan

# Keep kernel folder lean: Kaggle runtime receives `script.py`, and project code
# is delivered via `project.zip` from dataset source.
if (Test-Path "kaggle\kernel\src") { Remove-Item "kaggle\kernel\src" -Recurse -Force }
if (Test-Path "kaggle\kernel\configs") { Remove-Item "kaggle\kernel\configs" -Recurse -Force }

Copy-Item -Path requirements.txt -Destination kaggle\kernel\requirements.txt -Force

# Build fallback project bundle for Kaggle code dataset.
$bundle = Join-Path $repoRoot "kaggle\kernel\project.zip"
if (Test-Path $bundle) { Remove-Item $bundle -Force }
$bundlePaths = @("src", "configs", "requirements.txt", "kaggle\kernel\configs_to_run.txt")
Compress-Archive -Path $bundlePaths -DestinationPath $bundle -Force

# Keep a copy for Kaggle code dataset.
$codeDatasetDir = Join-Path $repoRoot "kaggle\code_dataset"
if (!(Test-Path $codeDatasetDir)) { New-Item -ItemType Directory -Path $codeDatasetDir | Out-Null }
Copy-Item -Path $bundle -Destination (Join-Path $codeDatasetDir "project.zip") -Force

# Build canonical data archive for Kaggle data dataset.
$dataRoot = Join-Path $repoRoot "runs\data"
$dataDatasetDir = Join-Path $repoRoot "kaggle\data_dataset"
if (!(Test-Path $dataDatasetDir)) { New-Item -ItemType Directory -Path $dataDatasetDir | Out-Null }
$dataZip = Join-Path $dataDatasetDir "data.zip"

if ($SkipDataZip) {
  Write-Host "Skipping data.zip build (--SkipDataZip)." -ForegroundColor Yellow
} else {
  if (!(Test-Path $dataRoot)) {
    throw "Missing runs\data. Generate/download datasets before syncing Kaggle bundle."
  }

  $rebuildDataZip = $true
  if (-not $ForceDataZip -and (Test-Path $dataZip)) {
    $zipTime = (Get-Item $dataZip).LastWriteTimeUtc
    $latestDataFile = Get-ChildItem -Path $dataRoot -Recurse -File | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
    if ($null -ne $latestDataFile -and $latestDataFile.LastWriteTimeUtc -le $zipTime) {
      $rebuildDataZip = $false
    }
  }

  if ($rebuildDataZip) {
    Write-Host "Building data.zip from runs\data ..." -ForegroundColor Cyan
    $staging = Join-Path $dataDatasetDir "_pack"
    if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
    New-Item -ItemType Directory -Path $staging | Out-Null
    Copy-Item -Path $dataRoot -Destination (Join-Path $staging "data") -Recurse -Force
    if (Test-Path $dataZip) { Remove-Item $dataZip -Force }
    Compress-Archive -Path (Join-Path $staging "data") -DestinationPath $dataZip -Force
    Remove-Item $staging -Recurse -Force
  } else {
    Write-Host "data.zip is up to date, skip rebuild." -ForegroundColor Green
  }

  # Remove unpacked folder from kaggle/data_dataset to prevent accidental folder uploads.
  if (Test-Path (Join-Path $dataDatasetDir "data")) { Remove-Item (Join-Path $dataDatasetDir "data") -Recurse -Force }
}

Write-Host "Done. Next commands:" -ForegroundColor Green
Write-Host "  kaggle datasets version -p kaggle\data_dataset -m `"data update`"" -ForegroundColor Green
Write-Host "  kaggle datasets version -p kaggle\code_dataset -m `"code update`"" -ForegroundColor Green
Write-Host "  kaggle kernels push -p kaggle\kernel" -ForegroundColor Green
