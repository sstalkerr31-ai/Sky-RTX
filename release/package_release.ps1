# Builds SkyRT.dll + SkyRT_Panel.exe and packs one ready-to-upload zip (GitHub Releases / Nexus Mods).
# Run from "x64 Native Tools Command Prompt for VS" (needs Python 3.8+ in PATH):
#   powershell -ExecutionPolicy Bypass -File release\package_release.ps1 -Version 0.20.0
param([string]$Version = '0.20.0')
$ErrorActionPreference = 'Stop'
$root  = Split-Path $PSScriptRoot -Parent
$layer = Join-Path $root 'layer'
$panel = Join-Path $root 'panel'
$dist  = Join-Path $root "dist\SkyRT-$Version"

Push-Location $layer
cmd /c build.bat
if ($LASTEXITCODE -ne 0) { Pop-Location; throw 'build.bat (SkyRT.dll) failed' }
Pop-Location

Push-Location $panel
cmd /c build_panel.bat
if ($LASTEXITCODE -ne 0) { Pop-Location; throw 'build_panel.bat (SkyRT_Panel.exe) failed' }
Pop-Location

if (Test-Path $dist) { Remove-Item $dist -Recurse -Force }
New-Item -ItemType Directory -Path $dist | Out-Null
Copy-Item (Join-Path $layer 'SkyRT.dll')              $dist
Copy-Item (Join-Path $layer 'VK_LAYER_SKYRT.json')    $dist
Copy-Item (Join-Path $panel 'dist\SkyRT_Panel.exe')   $dist
Copy-Item (Join-Path $PSScriptRoot 'README.txt')      $dist

$zip = Join-Path $root "dist\SkyRT-$Version.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path "$dist\*" -DestinationPath $zip
Write-Host "Release ready: $zip"
