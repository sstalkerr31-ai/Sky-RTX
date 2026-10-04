@echo off
setlocal
cd /d "%~dp0"
rem The layer only loads when SKYRT_ENABLE=1, and only for the process started from this file,
rem so other Vulkan programs are not affected.
if not exist SkyPath.txt (
  echo Drag Sky.exe onto this window, or paste its full path, then press Enter:
  set /p SKYEXE=
  echo %SKYEXE:"=%> SkyPath.txt
)
set /p SKYEXE=<SkyPath.txt
set SKYRT_ENABLE=1
for %%I in ("%SKYEXE%") do set SKYDIR=%%~dpI
cd /d "%SKYDIR%"
start "" "%SKYEXE%"
