# Configure + build inside the VS 2026 developer environment.
param([switch]$Clean, [switch]$Configure, [switch]$Run, [string]$Preset = 'msvc')
$vs = 'C:\Program Files\Microsoft Visual Studio\18\Community'
$cuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4'
Import-Module "$vs\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
$env:VCPKG_ROOT = "$vs\VC\vcpkg"
$env:CUDA_PATH = $cuda
$env:PATH = "$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;" +
            "$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;" +
            "$cuda\bin;" +
            "C:\Program Files (x86)\Microsoft Visual Studio\Installer;" + $env:PATH
Set-Location $PSScriptRoot
if ($Clean) { Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue }
$bin = if ($Preset -eq 'msvc') { 'build/msvc' } else { 'build/test' }
if ($Clean -or $Configure -or -not (Test-Path "$bin/build.ninja")) {
    cmake --preset $Preset
    if (-not $?) { exit 1 }
}
cmake --build --preset $Preset
if (-not $?) { exit 1 }
if ($Run) { & .\build\msvc\mandelgpu.exe }
