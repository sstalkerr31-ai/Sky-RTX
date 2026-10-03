# 🌌 SkyRT — Real-Time Ray Tracing Layer for Sky: Children of the Light

> **Status:** 🛠️ Active Development / Early Proof of Concept

An experimental, high-performance Vulkan Injection Layer that brings **real-time ray-traced shadows** and hardware-accelerated global lighting experiments to *Sky: Children of the Light* on PC.

---

### 🎥 Teaser / Preview
*(<img width="1279" height="1023" alt="Снимок экрана 2026-10-03 204409" src="https://github.com/user-attachments/assets/a6b61e9b-08b4-4eaa-89ef-316024ee28cf" />
)*

### ✨ Current Features
- 🚀 **Zero Engine Modification:** Works via custom Vulkan Layer interception.
- 📐 **Dynamic Acceleration Structures:** Reconstructs scene geometry into BLAS/TLAS on the fly.
- ☀️ **Ray-Queried Shadows:** Real-time shadow casting from directional light sources using native RT pipeline.
- ⚡ **Minimal Overhead:** Targeted 60+ FPS performance on modern RTX hardware.

### 🚧 Roadmap
- [x] Custom Vulkan Layer injection & HDR buffer writing
- [x] UBO parsing and View/Projection matrix extraction
- [x] BLAS/TLAS construction for static scene geometry
- [x] Working `ray_query` shadow compute pass
- [ ] Refine depth buffer reconstruction & camera matrices
- [ ] Support for dynamic/character geometries
- [ ] Denoising filter & GUI overlay control

---
*Created by [Stalker 31]*
