@echo off
setlocal
if "%VULKAN_SDK%"=="" (
  echo VULKAN_SDK is not set. Install the Vulkan SDK and reopen this prompt.
  exit /b 1
)
where cl >nul 2>nul
if errorlevel 1 (
  echo cl.exe not found. Run this from "x64 Native Tools Command Prompt for VS".
  exit /b 1
)
cl /nologo /LD /MD /O2 /Zi /std:c++17 /EHsc /W3 /I"%VULKAN_SDK%\Include" SkyRT.cpp /Fe:SkyRT.dll /link /DEBUG user32.lib
if errorlevel 1 exit /b 1
echo.
echo Built SkyRT.dll. Next: powershell -ExecutionPolicy Bypass -File install.ps1
