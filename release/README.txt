SkyRT - hardware ray tracing for Sky: Children of the Light (PC)
=================================================================

REQUIREMENTS
  Windows 10/11, a GPU with Vulkan ray_query support (tested on an RTX 5070 Ti only), up to date drivers.

INSTALL
  1. Unpack this folder somewhere permanent. Do NOT move it afterwards (the registry stores its path).
  2. Run Install.bat once.
  3. Start the game with Run_Sky_with_RT.bat (first time it asks for the path to Sky.exe).
     Steam / other launchers: set the environment variable SKYRT_ENABLE=1 for the game process.

CONTROLS
  Ctrl+Home  ray tracing on / off
  Ctrl+End   cycle debug views
  SkyRT.cfg (created next to SkyRT.dll on the first run) is re-read while the game runs.
  If the game looks wrong: set taa=0 or instgeo=0, or delete SkyRT.cfg to restore defaults.

UNINSTALL
  Run Uninstall.bat, then delete the folder.

LOGS
  Every run writes SkyRT_<date>_<pid>.log next to the DLL. Attach it to bug reports.

NOTES
  The mod does not change any game files. It is an unofficial fan project, not affiliated with
  thatgamecompany. Use at your own risk.
