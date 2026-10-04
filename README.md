# 🌌 SkyRT — Real-Time Ray Tracing Layer for *Sky: Children of the Light*

> **Status:** 🛠️ Active development / experimental. PC only, Vulkan, NVIDIA RTX (hardware `ray_query`).

An implicit **Vulkan layer** (`SkyRT.dll`) that adds hardware ray tracing to *Sky: Children of the Light* without modifying the game: it hooks the Vulkan loader, rebuilds the scene as acceleration structures on the fly and composites ray-traced lighting into the game's HDR image before post-processing.

### 🎥 Preview
<img width="1000" alt="SkyRT: RTX OFF / RTX ON" src="https://github.com/user-attachments/assets/2f8c7eaa-109c-4fa9-b472-e686fc27014d" />

---

## 🥇 Why SkyRT is different

As of October 2026, **no other publicly available project brings hardware ray tracing to *Sky: Children of the Light*** (searched GitHub, Nexus Mods and the web). What exists for Sky today is post-processing:

| | ReShade / "RTGI" presets | **SkyRT** |
|---|---|---|
| Rays traced against | the depth buffer (screen space) | **real scene geometry (BLAS/TLAS)** |
| Shadow from something off-screen or behind a wall | ❌ impossible | ✅ |
| Shadows of moving characters | ❌ | ✅ rebuilt every frame |
| Uses RTX hardware (`ray_query`) | ❌ | ✅ |
| Game files modified | no | no (implicit Vulkan layer) |

NVIDIA RTX Remix does not apply either: it targets DirectX 8/9 fixed-function games, not a modern Vulkan renderer like Sky's.

If you know of another hardware-RT project for Sky, open an issue and it will be listed here.

---

## ✨ What it does today

| Feature | State |
|---|---|
| Soft **ray-traced sun shadows** (penumbra from a finite sun size) | ✅ working |
| **Characters** cast correct, smooth shadows (CPU-skinned meshes get their own BLAS every frame) | ✅ working |
| Ray-traced **ambient occlusion** + one-bounce **GI** with colour bleeding | ✅ working |
| **Temporal accumulation + bilateral denoise** (reprojection, depth rejection, variance clipping) | ✅ working |
| Sun direction read from the game, with outlier rejection between scenes | ✅ working |
| Forced **anisotropic filtering** (16x by default) and optional LOD bias | ✅ working |
| A few instanced props baked into the BLAS (position + scale) | 🧪 experimental, small effect |
| Water / reflections | ❌ not yet: the water surface has not been located |
| Grass and foliage in the acceleration structure | ❌ not yet |
| Light from fires / emissive objects, multi-bounce GI | ❌ not yet |
| DLSS / upscaling | ❌ not planned for now (needs motion vectors, camera jitter, SDK) |

## 🧩 How it works (short)

- The layer enables `VK_KHR_acceleration_structure`, `VK_KHR_ray_query`, `VK_KHR_deferred_host_operations` and buffer device address on the game's device.
- Draw calls of the main pass are recorded (vertex/index buffers, pipeline, indirect args). Static geometry is cached in a ring of BLAS slots; geometry in CPU-written vertex buffers (skinned characters) is rebuilt every frame in a separate dynamic BLAS. Both go into one TLAS.
- After the main render pass two compute passes run: `rt.comp` (shadow, AO and bounce rays via `ray_query`) and `rt_fx.comp` (bilateral blur, temporal accumulation, composite into the colour image).
- Game facts the layer relies on: reverse-Z depth, y-up, a 1584-byte frame UBO (view-projection, camera, light direction).

## 🛠️ Build & install (Windows)

Requirements: Visual Studio (C++ x64), [Vulkan SDK](https://vulkan.lunarg.com/), an RTX GPU with up-to-date drivers.

1. Open **x64 Native Tools Command Prompt for VS** in the `layer/` folder.
2. `build.bat` → produces `SkyRT.dll`.
3. `powershell -ExecutionPolicy Bypass -File install.ps1` registers the implicit layer (use `-Machine` from an admin shell if needed, `-Uninstall` to remove).
4. Start the game with the environment variable `SKYRT_ENABLE=1`.

`rt_spv.h` / `rtfx_spv.h` are the compiled shaders. After editing a `.comp` file regenerate its header with `glslangValidator -V --target-env vulkan1.2`.

## 🎮 Controls & config

- **Ctrl+Home** — ray tracing on/off
- **Ctrl+End** — cycle debug views
- `SkyRT.cfg` next to the DLL is re-read while the game runs. Notable keys: `autosun`, `az`/`el`, `sunsize`, `aniso`, `lodbias`, `dyngeo`, `instgeo`, `instscale`, `taa`, `taan`.
- Each run writes a `SkyRT_*.log` (useful for bug reports).

## 🚧 Roadmap

- [x] Vulkan layer injection, HDR buffer access, UBO / matrix extraction
- [x] BLAS/TLAS for static geometry, `ray_query` shadows
- [x] Dynamic geometry (characters), smooth moving shadows
- [x] AO + bounce GI, denoiser, temporal accumulation
- [ ] Find and ray-trace the water surface (reflections / refractions)
- [ ] Vegetation in the acceleration structure
- [ ] Emissive / fire lights
- [ ] Optional launcher

## ⚠️ Disclaimer

Unofficial fan project, not affiliated with thatgamecompany. It only hooks the graphics API on your machine and modifies no game files. Use at your own risk, and not in situations where anti-cheat could object (the layer is opt-in via `SKYRT_ENABLE`).

---

## 🇷🇺 Коротко по-русски

Неявный Vulkan-слой с аппаратной трассировкой лучей для Sky: Children of the Light. Из найденного нами это первый открытый проект именно с аппаратными лучами по реальной геометрии сцены, а не ReShade-пост-обработкой по буферу глубины. Сейчас: мягкие RT-тени от солнца (в том числе от персонажей), AO и один отскок света, шумоподавление с накоплением по кадрам, анизотропная фильтрация. Вода, трава и свет от огня ещё не сделаны. Сборка: `build.bat` → `install.ps1` → запуск игры с `SKYRT_ENABLE=1`.

*Created by [Stalker 31]*
