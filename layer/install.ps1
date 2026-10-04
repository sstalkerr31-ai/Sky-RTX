param([switch]$Uninstall, [switch]$Machine)

# -Machine writes to HKLM (needs an admin PowerShell). Use it if the game runs elevated
# or the layer does not load from HKCU.
$manifest = Join-Path $PSScriptRoot 'VK_LAYER_SKYRT.json'
$root = if ($Machine) { 'HKLM:' } else { 'HKCU:' }
$key = "$root\SOFTWARE\Khronos\Vulkan\ImplicitLayers"

if ($Uninstall) {
    Remove-ItemProperty -Path $key -Name $manifest -ErrorAction SilentlyContinue
    Write-Host "Layer unregistered from $key"
    exit 0
}

if (-not (Test-Path (Join-Path $PSScriptRoot 'SkyRT.dll'))) {
    Write-Host "SkyRT.dll not found next to the manifest. Run build.bat first."
    exit 1
}

if (-not (Test-Path $key)) { New-Item -Path $key -Force | Out-Null }
New-ItemProperty -Path $key -Name $manifest -PropertyType DWord -Value 0 -Force | Out-Null
Write-Host "Layer registered in $key"
Write-Host "It loads only when SKYRT_ENABLE=1 is set in the game's environment."
Write-Host "Do not move this folder after installing (the registry stores the manifest path)."
