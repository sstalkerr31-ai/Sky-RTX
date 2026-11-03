@echo off
rem Build levelprobe.exe. Run from "x64 Native Tools Command Prompt for VS".
setlocal
cd /d "%~dp0"
cl /nologo /O2 /std:c++17 /EHsc /W3 /utf-8 levelprobe.cpp ..\..\third_party\meshoptimizer\vertexcodec.cpp /Fe:levelprobe.exe
