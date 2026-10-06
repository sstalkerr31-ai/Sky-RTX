@echo off
rem Builds SkyRT_Panel.exe (one file, no console). Needs Python 3.8+ in PATH.
python -m pip install --upgrade pyqt5 pyinstaller
if errorlevel 1 exit /b 1
python -m PyInstaller --noconsole --onefile --name SkyRT_Panel skyrt_panel.py
if errorlevel 1 exit /b 1
echo.
echo Done: dist\SkyRT_Panel.exe
echo Put SkyRT_Panel.exe next to SkyRT.dll and VK_LAYER_SKYRT.json, then run it.
