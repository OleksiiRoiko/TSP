param(
    [ValidateSet("build", "watch", "clean", "distclean")]
    [string]$Task = "build"
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    # Make MiKTeX/Perl available even if PATH is not refreshed yet.
    $extraPaths = @(
        (Join-Path $env:LOCALAPPDATA "Programs\\MiKTeX\\miktex\\bin\\x64"),
        "C:\\Strawberry\\perl\\bin",
        "C:\\Strawberry\\c\\bin"
    ) | Where-Object { Test-Path $_ }
    if ($extraPaths.Count -gt 0) {
        $env:Path = (($extraPaths -join ";") + ";" + $env:Path)
    }

    $latexmkExe = "latexmk"
    $latexmkCmd = Get-Command $latexmkExe -ErrorAction SilentlyContinue
    if (-not $latexmkCmd) {
        $fallback = Join-Path $env:LOCALAPPDATA "Programs\\MiKTeX\\miktex\\bin\\x64\\latexmk.exe"
        if (Test-Path $fallback) {
            $latexmkExe = $fallback
        }
        else {
            throw "latexmk not found. Install MiKTeX + Strawberry Perl first."
        }
    }

    switch ($Task) {
        "build" {
            & $latexmkExe -pdf main.tex
        }
        "watch" {
            & $latexmkExe -pvc -pdf main.tex
        }
        "clean" {
            & $latexmkExe -c
        }
        "distclean" {
            & $latexmkExe -C
            if (Test-Path build) {
                Remove-Item -Recurse -Force build
            }
        }
    }
}
finally {
    Pop-Location
}
