# Builds SkyRT.dll and packs a ready-to-upload release zip (Nexus Mods / GitHub Releases).
# Run from "x64 Native Tools Command Prompt for VS":  powershell -ExecutionPolicy Bypass -File release\package_release.ps1 -Version 0.20.0
param([string]$Version = '0.20.0')
$ErrorActionPreference = 'Stop'
$root  = Split-Path $PSScriptRoot -Parent
$layer = Join-Path $root 'layer'
$dist  = Join-Path $root "dist\SkyRT-$Version"

Push-Location $layer
cmd /c build.bat
if ($LASTEXITCODE -ne 0) { Pop-Location; throw 'build.bat failed' }
Pop-Location

if (Test-Path $dist) { Remove-Item $dist -Recurse -Force }
New-Item -ItemType Directory -Path $dist | Out-Null
Copy-Item (Join-Path $layer 'SkyRT.dll')           $dist
Copy-Item (Join-Path $layer 'VK_LAYER_SKYRT.json') $dist
Copy-Item (Join-Path $layer 'install.ps1')         $dist
foreach ($f in 'Install.bat','Uninstall.bat','Run_Sky_with_RT.bat','README.txt') { Copy-Item (Join-Path $PSScriptRoot $f) $dist }

$zip = Join-Path $root "dist\SkyRT-$Version.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path "$dist\*" -DestinationPath $zip
Write-Host "Release ready: $zip"
