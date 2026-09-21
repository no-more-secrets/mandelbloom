# Build the Release executable and package it with NSIS.
#   .\installer\build-installer.ps1              -> dist\Mandelbloom-Setup-<version>.exe
#   .\installer\build-installer.ps1 -TestMode    -> build\installer\Mandelbloom-Setup-test.exe (per-user, no UAC)
# The version comes from project(... VERSION x.y.z) in CMakeLists.txt.
# NSIS is used from build\tools\nsis-* (downloaded on first use).
param([switch]$TestMode, [switch]$SkipBuild)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent

$cm = Get-Content "$root\CMakeLists.txt" -Raw
if ($cm -notmatch 'project\(\s*\w+\s+VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)') { throw 'no VERSION in CMakeLists.txt' }
$version = $Matches[1]

# The icon master is docs\icon.png; src\icon.ico is derived from it.
$master = "$root\docs\icon.png"; $ico = "$root\src\icon.ico"
if ((Test-Path $master) -and (-not (Test-Path $ico) -or (Get-Item $master).LastWriteTime -gt (Get-Item $ico).LastWriteTime)) {
    if (Get-Command magick -ErrorAction SilentlyContinue) {
        & magick $master -define icon:auto-resize=256,128,64,48,32,16 $ico
        Write-Host 'icon rebuilt from docs\icon.png'
    } else { Write-Warning 'ImageMagick (magick) not found: src\icon.ico not refreshed from docs\icon.png' }
}

if (-not $SkipBuild) {
    & "$root\build.ps1" -Preset release
    if (-not $?) { exit 1 }
}
$exe = "$root\build\release\Mandelbloom.exe"
if (-not (Test-Path $exe)) { throw "missing $exe" }

$stage = "$root\build\installer\stage"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $stage | Out-Null
Copy-Item $exe $stage
Copy-Item "$root\build\release\*.dll" $stage
$crt = Get-ChildItem "C:\Program Files*\Microsoft Visual Studio\*\*\VC\Redist\MSVC\*\x64\Microsoft.VC*.CRT" -Directory |
    Sort-Object FullName -Descending | Select-Object -First 1
if (-not $crt) { throw 'VC++ runtime redistributable folder not found' }
foreach ($d in 'msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll') {
    Copy-Item (Join-Path $crt.FullName $d) $stage
}
Copy-Item "$root\README.md" "$stage\README.txt"

$makensis = Get-ChildItem "$root\build\tools\nsis-*\makensis.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $makensis) {
    $tools = "$root\build\tools"
    New-Item -ItemType Directory -Force $tools | Out-Null
    $zip = "$tools\nsis.zip"
    Write-Host 'downloading NSIS 3.11'
    Invoke-WebRequest -UseBasicParsing -Uri 'https://sourceforge.net/projects/nsis/files/NSIS%203/3.11/nsis-3.11.zip/download' -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $tools -Force
    $makensis = Get-ChildItem "$tools\nsis-*\makensis.exe" | Select-Object -First 1
}

New-Item -ItemType Directory -Force "$root\dist" | Out-Null
$out = if ($TestMode) { "$root\build\installer\Mandelbloom-Setup-test.exe" } else { "$root\dist\Mandelbloom-Setup-$version.exe" }
$nsisArgs = @("/DVERSION=$version", "/DSTAGE=$stage", "/DOUTFILE=$out")
if ($TestMode) { $nsisArgs += '/DTESTMODE' }
& $makensis.FullName /V2 @nsisArgs "$root\installer\mandelbloom.nsi"
if (-not $?) { exit 1 }
Write-Host "installer: $out"
