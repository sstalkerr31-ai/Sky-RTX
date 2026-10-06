**English** | [Русский](README.ru.md)

# 🌌 SkyRT — Hardware Ray Tracing for *Sky: Children of the Light*

> **Status:** 🛠️ experimental, actively developed. PC only · Vulkan · NVIDIA RTX (hardware `ray_query`).

SkyRT is an **implicit Vulkan layer** that adds real hardware ray tracing to *Sky: Children of the Light* **without modifying any game file**. It rebuilds the scene as acceleration structures every frame, traces rays against the real geometry and composites the result into the game's HDR image before post-processing. It comes with a small **control panel** (one-click install, live settings, game launcher).

### 🎥 Preview
<img width="1000" alt="SkyRT: RTX OFF / RTX ON" src="https://github.com/user-attachments/assets/2f8c7eaa-109c-4fa9-b472-e686fc27014d" />

---

## Contents
1. [Features](#-features)
2. [Why it is different](#-why-skyrt-is-different)
3. [Quick start (3 steps)](#-quick-start)
4. [The control panel](#-the-control-panel)
5. [Settings reference](#-settings-reference)
6. [In-game hotkeys](#-in-game-hotkeys)
7. [Troubleshooting & FAQ](#-troubleshooting--faq)
8. [Build from source](#-build-from-source)
9. [How it works](#-how-it-works)
10. [Roadmap & limitations](#-roadmap--limitations)
11. [Disclaimer & license](#-disclaimer--license)

## ✨ Features

| Feature | State |
|---|---|
| Soft **ray-traced sun shadows** (penumbra from a finite sun size) | ✅ |
| **Characters cast smooth, correct shadows** (skinned meshes get their own BLAS every frame) | ✅ |
| Ray-traced **ambient occlusion** + one-bounce **GI** with colour bleeding | ✅ |
| **Denoising**: bilateral blur + temporal accumulation (reprojection, depth rejection, variance clipping) | ✅ |
| Sun direction read from the game, with outlier rejection between scenes | ✅ |
| Forced **anisotropic filtering** (16x) and optional LOD bias | ✅ |
| One-click installer + live control panel | ✅ |
| A few instanced props in the acceleration structure | 🧪 experimental, small effect |
| Water / reflections | ❌ not yet (the water surface has not been located) |
| Grass and foliage in the acceleration structure | ❌ not yet |
| Light from fires / emissive objects, multi-bounce GI | ❌ not yet |

## 🥇 Why SkyRT is different

As of October 2026 we know of **no other public project that brings hardware ray tracing to Sky** (searched GitHub, Nexus Mods and the web). What exists today is post-processing:

| | ReShade / "RTGI" presets | **SkyRT** |
|---|---|---|
| Rays are traced against | the depth buffer (screen space) | **real scene geometry (BLAS/TLAS)** |
| Shadow of something off-screen or behind a wall | ❌ | ✅ |
| Shadows of moving characters | ❌ | ✅ |
| Uses RTX hardware (`ray_query`) | ❌ | ✅ |
| Game files modified | no | no |

NVIDIA RTX Remix does not apply: it targets DirectX 8/9 fixed-function games. If you know of another hardware-RT project for Sky, please open an issue.

## 🚀 Quick start

**Requirements:** Windows 10/11 · an NVIDIA RTX GPU with up-to-date drivers (tested only on an RTX 5070 Ti; any GPU exposing `VK_KHR_ray_query` + `VK_KHR_acceleration_structure` should work in theory) · *Sky: Children of the Light* for PC.

1. **Download** the latest `SkyRT-x.y.z.zip` from the [Releases](../../releases) page and unpack it anywhere.
   *(No release yet? See [Build from source](#-build-from-source).)*
2. **Run `SkyRT_Panel.exe`** → tab **Game** → press **Install ray tracing**. This copies the layer to `%LOCALAPPDATA%\SkyRT` and registers it for your Windows user. Admin rights are not needed. After this you can delete the unpacked folder.
3. **Set the path to `Sky.exe`** (button *Browse*) and press **▶ Play with rays**.

That's it. In game press **Ctrl+Home** to compare with/without rays.

> The layer loads only for the game process started with the panel's *Play with rays* button (it sets `SKYRT_ENABLE=1` for that process). Other Vulkan programs are not affected.
> If you must start the game through another launcher (Steam, a custom launcher), set the environment variable `SKYRT_ENABLE=1` for that launcher.

**Uninstall:** open the panel → **Uninstall**. Then delete `%LOCALAPPDATA%\SkyRT` if you want to remove the files too.

## 🎛️ The control panel

| Tab | What it does |
|---|---|
| **Game** | Status of the files / layer / game path · *Install* / *Uninstall* · path to `Sky.exe` · **Play with rays** · Play without rays · language switch (RU/EN) |
| **Graphics** | Presets (Low / Medium / High / Ultra / Custom) and sliders for every setting. Every change is written to `SkyRT.cfg` and applied **live** while the game runs |
| **Log** | The newest `SkyRT_*.log` with auto-refresh and *Open folder*. Attach this file to bug reports |

Presets: **Low** 1 shadow ray / 2 AO rays, no GI · **Medium** 2 / 4 (default) · **High** 4 / 6, GI range 40 · **Ultra** 8 / 8, 24 accumulated frames, GI range 60.

## ⚙️ Settings reference

Everything the panel changes lives in `%LOCALAPPDATA%\SkyRT\SkyRT.cfg` (plain `key=value` lines, re-read about once a second). You can edit it by hand.

| Key | Range | Meaning |
|---|---|---|
| `enabled` | 0/1 | Ray tracing on/off (0 = original picture) |
| `shadows` · `strength` · `sunsize` · `shrays` | 0/1 · 0–1 · 0–10° · 1–8 | Shadows: on/off, darkness, penumbra size, rays per pixel |
| `ao` · `aostrength` · `aoradius` · `aorays` | 0/1 · 0–1 · 0.05–20 · 1–8 | Ambient occlusion: on/off, strength, reach (world units), rays per pixel |
| `gi` · `gistrength` · `girange` | 0/1 · 0–2 · 1–200 | Bounce light: on/off, strength, how far rays look |
| `taa` · `taan` | 0/1 · 1–64 | Temporal accumulation on/off, max accumulated frames (lower = less ghosting, more noise) |
| `dyngeo` | 0/1 | Characters / animated meshes cast ray-traced shadows |
| `instgeo` | 0/1 | Small instanced props in the acceleration structure (experimental) |
| `aniso` · `lodbias` | 0–16 · −2…1 | Anisotropic filtering, texture sharpness shift (**applied on the next game start**) |
| `autosun` · `az` · `el` | 0/1 · 0–360° · 1–89° | Take the sun from the game, or set it manually |
| `view` | 0–7 | Debug views (see hotkeys) |

More rays = less noise, more GPU work. Sky is usually CPU-bound, so the GPU normally has plenty of headroom.

## ⌨️ In-game hotkeys

| Keys | Action |
|---|---|
| **Ctrl+Home** | Ray tracing on/off (compare with the original picture) |
| **Ctrl+End** | Next view: 0 full · 1 shadows only · 2 AO only · 3 bounce light only · 4 AO map (debug) · 5 bounce map (debug) · 6 grid (debug) · 7 normals (debug) |

If the picture looks strange, press **Ctrl+End** until view 0 is back (or set `view=0`).

## 🧯 Troubleshooting & FAQ

**Nothing changes / looks like the original.**
Check the panel: *Vulkan layer* must say *installed*, and the game must be started with **Play with rays**. Open the **Log** tab: a fresh `SkyRT_*.log` should appear when the game starts. No log = the layer did not load (wrong launcher, `SKYRT_ENABLE` not set).

**The game does not start from the panel.** The panel starts `Sky.exe` directly. If your setup needs Steam/another launcher, start the game there with `SKYRT_ENABLE=1` set in the environment.

**Weird colours / black world.** Press Ctrl+End until view 0, or set `view=0`. If it persists, set `enabled=0` and send the log.

**Low FPS.** Use the *Low* preset, or lower `shrays` / `aorays`; try `taa=1`. Disable `instgeo` and `dyngeo` to test.

**Ghost trails behind characters / fast camera.** Lower `taan` (try 6).

**Shadows jump when the scene changes.** The sun direction is smoothed; a real scene change takes about a second to blend.

**Crash or freeze.** Please open an issue with the newest `SkyRT_*.log` (Log tab → *Open folder*). A missing "clean shutdown" line at the end of the log means the game crashed.

**Windows SmartScreen / antivirus warns about `SkyRT_Panel.exe` / `SkyRT.dll`.** The files are not code-signed, and DLLs that hook graphics APIs are often flagged by heuristics. You can build everything yourself from this repository.

**Is it safe for my account?** The layer does not touch game files or memory and sends nothing anywhere, but it is an unofficial modification of a client of an online game. Use at your own risk.

**Does it work on AMD / Intel?** Untested. In theory any GPU with `VK_KHR_ray_query` and `VK_KHR_acceleration_structure` can run it.

## 🛠️ Build from source

Requirements: Visual Studio (C++ x64) · [Vulkan SDK](https://vulkan.lunarg.com/) · Python 3.8+ (for the panel).

```bat
:: open "x64 Native Tools Command Prompt for VS", then:
cd layer
build.bat                       :: -> SkyRT.dll
cd ..\panel
build_panel.bat                 :: -> dist\SkyRT_Panel.exe  (installs PyQt5 + PyInstaller)
cd ..
powershell -ExecutionPolicy Bypass -File release\package_release.ps1 -Version 0.20.0   :: -> dist\SkyRT-0.20.0.zip
```

Run the panel from source: `pip install PyQt5` then `python panel\skyrt_panel.py` (put `SkyRT.dll` and `VK_LAYER_SKYRT.json` next to it).
Manual install without the panel: `layer\install.ps1` registers the layer (`-Uninstall` removes it; `-Machine` from an admin shell writes to HKLM).

After editing a `.comp` shader: `glslangValidator -V --target-env vulkan1.2 layer\rt_fx.comp -o rtfx.spv`, then `python tools\spv2h.py rtfx.spv layer\rtfx_spv.h kRtFxSpv rt_fx.comp` (the trace shader is `rt.comp` → `rt_spv.h`, array `kRtSpv`).

## 🧩 How it works

- The layer enables `VK_KHR_acceleration_structure`, `VK_KHR_ray_query`, `VK_KHR_deferred_host_operations` and buffer device address on the game's device.
- Draw calls of the main pass are recorded (vertex/index buffers, pipeline, indirect arguments). Static geometry is cached in a ring of BLAS slots; geometry in CPU-written vertex buffers (skinned characters) and a few instanced props are rebuilt every frame into a dynamic BLAS. Both go into one TLAS.
- After the main render pass two compute passes run: `rt.comp` (shadow, AO and bounce rays through `ray_query`) and `rt_fx.comp` (bilateral blur, temporal accumulation, composite into the colour image).
- Game facts the layer relies on: reverse-Z depth, y-up, a 1584-byte frame UBO (view-projection, camera, light direction).

## 🚧 Roadmap & limitations

- [x] Vulkan layer, HDR/depth access, matrix extraction
- [x] BLAS/TLAS, `ray_query` shadows, dynamic geometry (characters)
- [x] AO + bounce GI, denoiser, temporal accumulation
- [x] One-click installer + control panel
- [ ] Find and ray-trace the water surface (reflections / refractions)
- [ ] Vegetation in the acceleration structure
- [ ] Multi-bounce GI, light from fires and emissive objects

Known limitations: static geometry is assumed not to move; GPU-animated objects (wind, waves) are not reflected in shadows; hit surfaces have no material, so bounce light is taken from what is visible on screen.

## ⚠️ Disclaimer & license

Unofficial fan project, not affiliated with thatgamecompany. It only hooks the graphics API on your PC and modifies no game files. Use at your own risk. Source code under the [MIT license](LICENSE).

*Created by [Stalker 31]*
