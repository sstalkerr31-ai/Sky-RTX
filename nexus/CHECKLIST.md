# Publishing on Nexus Mods

1. Build: `release\package_release.ps1 -Version 0.20.0` -> `dist\SkyRT-0.20.0.zip`
2. Test the zip on a clean folder: SkyRT_Panel.exe -> Install ray tracing -> Play with rays -> check `SkyRT_*.log` (Log tab).
3. Nexus: Sky: Children of the Light -> Upload mod -> category "Visuals and Graphics".
4. Paste `nexus/DESCRIPTION.bbcode` into the description, upload 3-5 screenshots (RTX OFF/ON side by side),
   put `nexus/CHANGELOG.txt` into the changelog field.
5. Permissions: pick the licence you want (the repo is MIT). Mention the GitHub source link.
6. Tick "this mod does not contain adult content"; no other requirements.
7. Be honest in the description about what is not done (water, foliage): users forgive limits, not surprises.
