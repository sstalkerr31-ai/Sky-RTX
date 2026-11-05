// levelprobe: loads a level's terrain with the layer's own reader and prints what the ray scene would get.
//   levelprobe <game folder> <level> [x y z]   (x y z: a point to check, e.g. "Starting avatar at" from Sky.log)
#include "../../layer/level_terrain.h"
#include "../../layer/level_models.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

int wmain(int argc, wchar_t** argv) {
    if (argc < 3) { fwprintf(stderr, L"usage: levelprobe <game folder> <level> [x y z]\n"); return 64; }
    std::wstring file;
    if (!skylevel::FindLevelFile(argv[1], argv[2], file)) { fwprintf(stderr, L"level %ls not found\n", argv[2]); return 1; }
    LARGE_INTEGER f, t0, t1;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t0);
    skylevel::Terrain t;
    std::string err;
    const bool ok = skylevel::LoadTerrain(file, t, &err);
    QueryPerformanceCounter(&t1);
    if (!ok) { fprintf(stderr, "failed: %s\n", err.c_str()); return 1; }
    printf("%ls: v%u verts=%zu tris=%zu patches %u (clouds left out: %u), %.1f ms\n", argv[2], t.version, t.pos.size() / 3,
           t.idx.size() / 3, t.patches, t.cloudPatches, 1000.0 * double(t1.QuadPart - t0.QuadPart) / double(f.QuadPart));
    printf("  min %.1f %.1f %.1f  max %.1f %.1f %.1f\n", t.mn[0], t.mn[1], t.mn[2], t.mx[0], t.mx[1], t.mx[2]);
    {
        LARGE_INTEGER m0, m1; QueryPerformanceCounter(&m0);
        skylevel::Models md;
        std::string me;
        const bool mok = skylevel::LoadModels(argv[1], argv[2], md, &me);
        QueryPerformanceCounter(&m1);
        printf("models: %s placed %u, %zu tris, %zu verts, %.1f ms | skipped: hidden %u, disabled %u, shader %u, skinned %u, missing %u, broken %u\n", mok ? "ok" : me.c_str(),
               md.placed, md.idx.size() / 3, md.pos.size() / 3, 1000.0 * double(m1.QuadPart - m0.QuadPart) / double(f.QuadPart), md.skippedHidden, md.skippedDisabled, md.skippedShader, md.skippedBones, md.missing, md.broken);
        if (mok) printf("  bounds %.1f %.1f %.1f .. %.1f %.1f %.1f\n", md.mn[0], md.mn[1], md.mn[2], md.mx[0], md.mx[1], md.mx[2]);
    }
    if (argc >= 6) {
        const float x = float(_wtof(argv[3])), y = float(_wtof(argv[4])), z = float(_wtof(argv[5]));
        // the terrain height under the point: highest triangle below it, or nearest above
        float best = NAN;
        for (size_t i = 0; i < t.idx.size(); i += 3) {
            const float* a = &t.pos[t.idx[i] * 3]; const float* b = &t.pos[t.idx[i + 1] * 3]; const float* c = &t.pos[t.idx[i + 2] * 3];
            const float d = (b[2] - c[2]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[2] - c[2]);
            if (std::fabs(d) < 1e-9f) continue;
            const float l1 = ((b[2] - c[2]) * (x - c[0]) + (c[0] - b[0]) * (z - c[2])) / d;
            const float l2 = ((c[2] - a[2]) * (x - c[0]) + (a[0] - c[0]) * (z - c[2])) / d;
            const float l3 = 1.0f - l1 - l2;
            if (l1 < 0 || l2 < 0 || l3 < 0) continue;
            const float h = l1 * a[1] + l2 * b[1] + l3 * c[1];
            if (std::isnan(best) || std::fabs(h - y) < std::fabs(best - y)) best = h;
        }
        if (std::isnan(best)) printf("  point %.2f %.2f %.2f: no terrain above or below\n", x, y, z);
        else printf("  point %.2f %.2f %.2f: terrain at y=%.2f (%.2f from the point)\n", x, y, z, best, y - best);
    }
    return 0;
}
