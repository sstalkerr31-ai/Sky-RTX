**English** | [Русский](README.ru.md)

# 🌌 SkyRT — Hardware Ray Tracing for *Sky: Children of the Light*

> **Status:** 🛠️ experimental, actively developed. PC only · Vulkan · NVIDIA RTX (hardware `ray_query`). Tested on one PC only (RTX 5070 Ti).
>
> **This is not full path tracing yet.** Rays now know the colour of off-screen surfaces and light them with one bounce, see [Path tracing: where we really are](#-path-tracing-where-we-really-are).

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
9. [Path tracing: where we really are](#-path-tracing-where-we-really-are)
10. [Architecture](#-architecture)
11. [Roadmap & limitations](#-roadmap--limitations)
12. [Disclaimer & license](#-disclaimer--license)

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
| Water and ice reflections | 🧪 experimental: the shaders are identified and the reflection is written, but it has never been confirmed on real water |
| Grass in the acceleration structure (shadows, AO) | 🧪 experimental |
| **Materials for rays**: geometry is linked to its colour (textures or per-draw colour) | 🧪 new, off by default (`matalbedo=1`) |
| **Lighting at the ray hit**: off-screen hits get sun, N·L, a shadow ray and shade | 🧪 new, one bounce, off by default |
| Light from fires, candles and lamps with ray-traced shadows | ✅ (new, being tuned) |
| Multi-bounce GI | ✅ (new, being tuned) |

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

> **About the defaults.** The default values were set by the developer to match their own taste, screen and PC. They are a starting point, not a rule: change any of them to whatever looks best to you (the panel sliders, or the file by hand). The mod is still in active development and the settings may change between versions.

| Key | Range | Meaning |
|---|---|---|
| `enabled` | 0/1 | Ray tracing on/off (0 = original picture) |
| `shadows` · `strength` · `sunsize` · `shrays` | 0/1 · 0–1 · 0–10° · 1–8 | Shadows: on/off, darkness, penumbra size, rays per pixel |
| `ao` · `aostrength` · `aoradius` · `aorays` | 0/1 · 0–1 · 0.05–20 · 1–8 | Ambient occlusion: on/off, strength, reach (world units), rays per pixel |
| `gi` · `gistrength` · `girange` | 0/1 · 0–2 · 1–200 | Bounce light: on/off, strength, how far rays look |
| `taa` · `taan` | 0/1 · 1–64 | Temporal accumulation on/off, max accumulated frames (lower = less ghosting, more noise) |
| `dyngeo` | 0/1 | Characters / animated meshes cast ray-traced shadows |
| `instgeo` | 0/1 | Small instanced props in the acceleration structure (experimental) |
| `deep` | 0/1 | **Deep bounce mode**: every ray path bounces several times, with exact surface normals of what it hits. Needs a GPU with ray tracing position fetch (RTX 30xx and newer; the log says if it is missing). Very heavy, the GPU heats up on purpose. Hotkey Ctrl+] |
| `bounces` | 1..8 | Bounces per path in deep mode (default 3). Cost grows with it. Hotkeys Ctrl+Up / Ctrl+Down |
| `pathtrace` | 0/1 | Screenshot mode: 4x rays per pixel, up to 256 frames of accumulation, deeper bounces. Despite the key name it adds no materials by itself. Hotkey Ctrl+Backspace |
| `ptadapt` | 0/1 | 1 = screenshot mode runs at full quality only while the camera stands still; 0 (default) = always full |
| `grass` | 0/1 | Grass in the acceleration structure: casts shadows and takes part in AO (experimental) |
| `gimulti` | 0–0.9 | Multi-bounce GI: light bounces more than once (every frame adds a bounce, needs `taa=1`); 0 = single bounce |
| `light` | 0/1 | Light from fire, candles and lamps: very bright pixels of the frame become light sources that light and shadow their surroundings (new, needs testing) |
| `lightstrength` · `lightrange` | 0–200 (default 2) · 1–60 | How strong the light is, how far (world units) one source reaches |
| `lightmax` | 0.05–5 | Soft ceiling of the light added to one pixel (lower = never blown out) |
| `lightthr` · `lightrays` · `lightdebug` | 0.3–50 · 1–4 · 0/1 | Brightness above which a pixel counts as a source (the log prints the brightest pixel it saw), shadow rays per pixel, paint detected sources magenta |
| `matalbedo` | 0/1 | **Materials and lighting at the ray hit.** Must be 1 in the file **before the game starts** (textures are tracked from the first frame); after that Ctrl+F9 toggles it live. No panel control yet |
| `matgain` | 0–8 | Brightness of light from a hit surface (default 1) |
| `texprobe` | 0/1 | Log only: which textures every shader pair samples. Set before the game starts |
| `waterfx` · `waterrefl` · `glint` | 0/1 · strength · 0/1 | Experimental water and ice: reflections on/off, reflection strength, sun glints |
| `aniso` · `lodbias` | 0–16 · −2…1 | Anisotropic filtering, texture sharpness shift (**applied on the next game start**) |
| `autosun` · `az` · `el` | 0/1 · 0–360° · 1–89° | Take the sun from the game, or set it manually |
| `view` | 0–9 | Debug views. 0 = normal, 5 = bounce light only, 8 = material colour of every pixel (black = no material), 9 = bounce light with every ray that uses a material painted red |

More rays = less noise, more GPU work. Sky is usually CPU-bound, so the GPU normally has plenty of headroom.

## ⌨️ In-game hotkeys

| Keys | Action |
|---|---|
| **Ctrl+Home** | Ray tracing on/off (compare with the original picture) |
| **Ctrl+End** | Debug: shader census. Press once to start recording the main pass, press again to print every shader pair with draw counts to the log (used to find the water / ice shaders). The old debug views are selected with `view=N` in the cfg |
| **Ctrl+]** | Deep bounce mode on/off (very heavy) |
| **Ctrl+Up / Ctrl+Down** | Bounces +1 / -1 (1..8) in deep mode |
| **Ctrl+Backspace** | Screenshot mode on/off: 4x rays, long accumulation. Stand still for a clean picture |
| **Ctrl+F9** | Materials on/off. Needs `matalbedo=1` when the game starts. The effect can be subtle: stand still and compare two screenshots |

If the picture looks strange, set `view=0`.

## 🧯 Troubleshooting & FAQ

**Nothing changes / looks like the original.**
Check the panel: *Vulkan layer* must say *installed*, and the game must be started with **Play with rays**. Open the **Log** tab: a fresh `SkyRT_*.log` should appear when the game starts. No log = the layer did not load (wrong launcher, `SKYRT_ENABLE` not set).

**The game does not start from the panel.** The panel starts `Sky.exe` directly. If your setup needs Steam/another launcher, start the game there with `SKYRT_ENABLE=1` set in the environment. In Steam launch options use `cmd /c "set SKYRT_ENABLE=1&& %command%"`, with no space before `&&` (the space would become part of the value).

**Weird colours / black world.** Set `view=0`. If it persists, set `enabled=0` and send the log.

**Low FPS.** Use the *Low* preset, or lower `shrays` / `aorays`; try `taa=1`. Disable `instgeo` and `dyngeo` to test.

**Ghost trails behind characters / fast camera.** Lower `taan` (try 6).

**Fire light is missing, or ordinary surfaces glow.** Set `lightdebug=1`: pixels treated as light sources turn magenta. If nothing is magenta at a fire, lower `lightthr`; if sunlit ground is magenta, raise it. `SkyRT_*.log` prints the brightest pixel value every few seconds (`rt: emissive lights: ...`).

**Ctrl+F9 changes nothing visible.** Check the log for `rt: KEY Ctrl+F9` and `rt: material coverage`. Then set `view=9`: if the world turns red, materials reach the rays and the effect is simply small in this scene. It shows most where the off-screen surroundings differ in colour from what you see.

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

After editing a `.comp` shader: `glslangValidator -V --target-env vulkan1.2 layer\rt_fx.comp -o rtfx.spv`, then `python tools\spv2h.py rtfx.spv layer\rtfx_spv.h kRtFxSpv rt_fx.comp`.

| Source | Header | Array | Notes |
|---|---|---|---|
| `rt.comp` | `rt_spv.h` | `kRtSpv` | trace pass for GPUs without position fetch |
| `rt.comp` with `-DPOSFETCH` | `rtpf_spv.h` | `kRtPfSpv` | the same pass with real hit normals |
| `rt_mat.comp` | `rtmat_spv.h` | `kRtMatSpv` | average colour of a material |
| `rt_deep.comp` | `rtdeep_spv.h` | `kRtDeepSpv` | deep bounce variant |
| `rt_fx.comp` | `rtfx_spv.h` | `kRtFxSpv` | denoise and composite |
| `rt_light.comp` | `rtlight_spv.h` | `kRtLightSpv` | emissive light search |
| `rt_lmerge.comp` | `rtlmerge_spv.h` | `kRtLMergeSpv` | persistent light list |

Regenerate **both** `rt_spv.h` and `rtpf_spv.h` after editing `rt.comp`. Use the sources in this repository: a shader must match its header before you regenerate it.

## 🧭 Path tracing: where we really are

SkyRT does not replace the game's renderer. It traces rays against the real scene and adds light to the game's own image.

| Stage | State |
|---|---|
| Rays against real geometry | ✅ |
| Hit on screen: colour read from the frame | ✅ |
| Hit off screen: the surface's real colour | 🧪 one average colour per material or draw, no UV lookup |
| Light at the hit: sun, N·L, shadow ray | 🧪 one bounce, in `rt.comp` |
| Light at the hit: ambient | 🧪 simplified to one shade colour, the game uses a cubemap |
| Materials in the deep bounce mode | ❌ `rt_deep.comp` still uses the old fallback |
| Characters and animated meshes as bounce surfaces | ❌ no material yet |
| Replacing the game's own lighting | ❌ not planned yet |

With materials on, the first measured result: ground next to blue rock reads about 25% bluer, and the walls lose some false light. About 55–60% of cached triangles had a material in the tested scenes.

## 🧩 Architecture

- The layer enables `VK_KHR_acceleration_structure`, `VK_KHR_ray_query`, `VK_KHR_deferred_host_operations` and buffer device address on the game's device.
- Draw calls of the main pass are recorded (vertex/index buffers, pipeline, indirect arguments). Static geometry is cached in a ring of BLAS slots; geometry in CPU-written vertex buffers (skinned characters) and a few instanced props are rebuilt every frame into a dynamic BLAS. Both go into one TLAS.
- After the main render pass four compute passes run: `rt_light.comp` (finds bright, emissive pixels per screen tile), `rt_lmerge.comp` (merges them into a small persistent list of light sources that fade in and out, so the light does not flicker), `rt.comp` (shadow, AO and bounce rays through `ray_query`) and `rt_fx.comp` (bilateral blur, temporal accumulation, composite into the colour image).
- **Trace pipeline choice:** with `VK_KHR_ray_tracing_position_fetch` the trace pass uses `rtpf_spv.h` and gets real triangle normals at the hit; without it the pass uses `rt_spv.h` and assumes N·L = 0.5. The second path has never been run.
- **Material tracking** (only with `matalbedo=1` at start): the layer records which descriptor sets and dynamic offsets each draw binds. Each cached geometry gets a material id, found in one of two ways:
  - *Texture materials* (`MeshSh`, `MeshShSl`, `MeshSlDarkvine`, table `kMatFs`): colour = texture `s1.b8` × `s1.b9`. `rt_mat.comp` averages the pair once, in the frame that drew it, so the image views are alive.
  - *Per-draw colour* (`RockFace*`, `GrassSh`, `SandSh`, table `kMatVs`): these shaders have no colour texture. The colour is a `vec3` in the per-draw block (set 0, binding 0), read from mapped memory at record time. Rock and grass are confirmed from the decompiled vertex shaders; the two sand offsets are not traced to the final colour yet.
- **Geometry → material table:** an SSBO with one section per AS slot, indexed by `slot × 30000 + geometry index`. It is written when a slot is rebuilt, and that slot is not in flight. Colours live in a second SSBO with a 256-byte stride.
- **Lighting at the hit** (`rt.comp`): on-screen hits keep the frame colour. An off-screen hit with a material gets `albedo × (sun colour × N·L × shadow ray + shade) × matgain`, using the game's own frame UBO values (sun colour at +912, shade at +944). The rock shader's own formula has the same shape, so the scale matches the frame.
- **Without a material** the old fallback stays: half of the current pixel's colour.
- **Composite** (`rt_fx.comp`): `colour += surface tint × bounce`, before the game's tone mapping and bloom.
- Game facts the layer relies on: reverse-Z depth, y-up, a 1584-byte frame UBO (view-projection, camera, light direction, sun colour). Shaders are identified by FNV-1a hashes of their SPIR-V, so a game update can silently drop them from the tables.

## 🚧 Roadmap & limitations

- [x] Vulkan layer, HDR/depth access, matrix extraction
- [x] BLAS/TLAS, `ray_query` shadows, dynamic geometry (characters)
- [x] AO + bounce GI, denoiser, temporal accumulation
- [x] One-click installer + control panel
- [x] Multi-bounce GI, light from fires and emissive objects (first version, being tuned)
- [x] Locate the water and ice shaders, first reflection code
- [x] Grass in the acceleration structure (experimental)
- [x] Link geometry to materials, light off-screen hits (first version)
- [ ] Confirm water and ice reflections on real water
- [ ] Real ambient at the hit (the game's cubemap)
- [ ] UV lookup at the hit instead of one colour per material
- [ ] More shaders in the material tables, per-draw tint for `MeshShSl` / `MeshSlDarkvine`
- [ ] Materials in the deep bounce mode
- [ ] Real light sources from the game instead of bright pixels (a possible clustered light list is under investigation)

Known limitations: static geometry is assumed not to move; GPU-animated objects (wind, waves) are not reflected in shadows; effects are inserted before the game's tone mapping and bloom; transparency (clouds, water, particles) is not solved; a game update can change shader hashes.

`DEVICE_LOST` on Alt+Enter has a written but unconfirmed fix.

## ⚠️ Disclaimer & license

Unofficial fan project, not affiliated with thatgamecompany. It only hooks the graphics API on your PC and modifies no game files. Use at your own risk. Source code under the [MIT license](LICENSE).

*Created by [Stalker 31]*
