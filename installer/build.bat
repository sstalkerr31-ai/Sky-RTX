@echo off
rem Builds the layer and the one-file installer. Needs only Visual Studio (C++): the Vulkan headers are in third_party\vulkan.
rem   set SKYRT_CHANNEL=test   which GitHub release the installer looks at for updates (default latest; GitHub's build sets it)
setlocal
where cl >nul 2>nul
if errorlevel 1 (
  echo cl.exe not found. Run this from "x64 Native Tools Command Prompt for VS".
  exit /b 1
)
cd /d "%~dp0"
if not exist build mkdir build
if "%SKYRT_CHANNEL%"=="" set SKYRT_CHANNEL=latest
if "%SKYRT_BUILD%"=="" set SKYRT_BUILD=local
set VKI=..\third_party\vulkan\include

rem the layer goes inside the exe; /MT so it needs no Visual C++ runtime on the player's PC
rem /Brepro: no build time inside the file, so the same code gives the same bytes and the installer can tell an install is current
cl /nologo /LD /MT /O2 /std:c++17 /EHsc /W3 /Brepro /DSKYRT_BUILD=\"%SKYRT_BUILD%\" /I%VKI% ..\layer\SkyRT.cpp /Fo:build\SkyRT.obj /Fe:build\SkyRT.dll /link /Brepro user32.lib
if errorlevel 1 exit /b 1

rc /nologo /fo build\app.res app.rc
if errorlevel 1 exit /b 1

set IM=third_party\imgui
cl /nologo /MT /O2 /std:c++17 /EHsc /W3 /utf-8 /Brepro /DSKYRT_CHANNEL=\"%SKYRT_CHANNEL%\" /I%IM% /I%IM%\backends /Ithird_party\stb /I%VKI% /Fo:build\ ^
  main.cpp %IM%\imgui.cpp %IM%\imgui_draw.cpp %IM%\imgui_tables.cpp %IM%\imgui_widgets.cpp %IM%\backends\imgui_impl_win32.cpp %IM%\backends\imgui_impl_dx11.cpp ^
  build\app.res /Fe:build\SkyRT_Setup.exe /link /Brepro /SUBSYSTEM:WINDOWS user32.lib gdi32.lib d3d11.lib dxgi.lib d3dcompiler.lib ole32.lib shell32.lib advapi32.lib comdlg32.lib uuid.lib winmm.lib winhttp.lib bcrypt.lib
if errorlevel 1 exit /b 1
rem the same program: a file name with "demo" in it changes nothing on the PC
copy /y build\SkyRT_Setup.exe build\SkyRT_Demo.exe >nul
echo.
echo Built installer\build\SkyRT_Setup.exe and SkyRT_Demo.exe (updates from the GitHub release "%SKYRT_CHANNEL%")
