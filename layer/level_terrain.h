// SkyRT level terrain reader: the ground of a whole level, straight from the game's BstBaked.meshes, so rays see terrain
// that is off the screen too. Only the GEO0 section is read (format versions 57+): one meshopt-compressed vertex buffer
// shared by all patches, a byte-triplet index buffer, then 56-byte patch records. Array A holds the terrain patches,
// array B the cloud patches, which are left out. Positions are already in game (world) space.
// Format knowledge: Sky-Export-Tool / ThatSkyOldServer SkyBstbake (MIT), see NOTICE. Decoder: meshoptimizer (MIT).
#pragma once
#include <windows.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "../third_party/meshoptimizer/meshoptimizer.h"

namespace skylevel {

struct Terrain {
    uint32_t version = 0;
    std::vector<float> pos;          // xyz per vertex, game space
    std::vector<uint32_t> color;     // RGBA8 baked vertex colour per vertex
    std::vector<uint32_t> matIds, matWeights;   // per vertex: 4 terrain material ids (bytes) and their weights (bytes, sum 255)
    std::vector<uint32_t> idx;       // triangles of the terrain patches only
    uint32_t patches = 0, cloudPatches = 0;
    float mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f};
};

// <game>\data\assets\<pack>\Data\Levels\<level>\BstBaked.meshes
inline bool FindLevelFile(const std::wstring& gameRoot, const std::wstring& level, std::wstring& out) {
    const std::wstring assets = gameRoot + L"\\data\\assets\\";
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW((assets + L"*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return false;
    bool found = false;
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) || fd.cFileName[0] == L'.') continue;
        std::wstring p = assets + fd.cFileName + L"\\Data\\Levels\\" + level + L"\\BstBaked.meshes";
        if (GetFileAttributesW(p.c_str()) != INVALID_FILE_ATTRIBUTES) { out = p; found = true; }
    } while (!found && FindNextFileW(h, &fd));
    FindClose(h);
    return found;
}

inline bool ReadFileAll(const std::wstring& path, std::vector<uint8_t>& data) {
    HANDLE f = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (f == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER sz;
    bool ok = GetFileSizeEx(f, &sz) && sz.QuadPart > 0 && sz.QuadPart < (1ll << 30);
    if (ok) {
        data.resize(size_t(sz.QuadPart));
        DWORD got = 0;
        ok = ReadFile(f, data.data(), DWORD(data.size()), &got, nullptr) && got == data.size();
    }
    CloseHandle(f);
    return ok;
}

// The file comes from the game and may change with an update: every offset is checked, a bad file just fails.
inline bool LoadTerrain(const std::wstring& path, Terrain& t, std::string* err = nullptr) {
    auto fail = [&](const char* m) { if (err) *err = m; return false; };
    std::vector<uint8_t> d;
    if (!ReadFileAll(path, d)) return fail("cannot read file");
    auto u32 = [&](size_t o) { uint32_t v; memcpy(&v, d.data() + o, 4); return v; };
    if (d.size() < 12 || memcmp(d.data(), "LVL0", 4) != 0) return fail("not LVL0");
    t = Terrain{};
    t.version = u32(4);
    if (t.version < 57) return fail("no GEO0 before version 57");
    const uint32_t entries = d[8];
    size_t go = 0, gl = 0;
    for (uint32_t i = 0; i < entries; ++i) {
        const size_t b = 12 + size_t(i) * 12;
        if (b + 12 > d.size()) return fail("bad table");
        if (memcmp(d.data() + b, "GEO0", 4) == 0) { go = u32(b + 4); gl = u32(b + 8); }
    }
    if (gl < 24 || go + gl > d.size()) return fail("no GEO0");
    const uint8_t* g = d.data() + go;
    uint32_t h[6];
    memcpy(h, g, 24);
    const uint32_t ibBytes = h[0], vcount = h[1], na = h[2], nb = h[3], vbSize = h[5];
    const size_t vbAt = 24, ibAt = vbAt + size_t(vbSize), patAt = ibAt + size_t(ibBytes);
    if (vcount == 0 || vcount > 16u << 20 || patAt + 56ull * (size_t(na) + nb) > gl) return fail("bad GEO0 header");

    std::vector<uint8_t> vb(size_t(vcount) * 36);
    if (meshopt_decodeVertexBuffer(vb.data(), vcount, 36, g + vbAt, vbSize) != 0) return fail("vertex decode failed");
    t.pos.resize(size_t(vcount) * 3);
    t.color.resize(vcount);
    t.matIds.resize(vcount);
    t.matWeights.resize(vcount);
    for (uint32_t v = 0; v < vcount; ++v) {
        memcpy(&t.pos[size_t(v) * 3], vb.data() + size_t(v) * 36, 12);
        memcpy(&t.matIds[v], vb.data() + size_t(v) * 36 + 16, 4);
        memcpy(&t.matWeights[v], vb.data() + size_t(v) * 36 + 20, 4);
        memcpy(&t.color[v], vb.data() + size_t(v) * 36 + 24, 4);
    }

    const uint8_t* ib = g + ibAt;
    t.patches = na;
    t.cloudPatches = nb;
    for (uint32_t p = 0; p < na; ++p) {
        const uint8_t* e = g + patAt + size_t(p) * 56;
        uint32_t ibo, vs; uint16_t ibl;
        memcpy(&ibo, e, 4); memcpy(&vs, e + 4, 4); memcpy(&ibl, e + 12, 2);
        if (size_t(ibo) + ibl > ibBytes) return fail("bad patch");
        for (uint32_t k = 0; k + 2 < ibl; k += 3) {
            const uint32_t a = vs + ib[ibo + k], b = vs + ib[ibo + k + 1], c = vs + ib[ibo + k + 2];
            if (a >= vcount || b >= vcount || c >= vcount) return fail("bad index");
            if (a == b || b == c || a == c) continue;
            t.idx.push_back(a); t.idx.push_back(b); t.idx.push_back(c);
            for (uint32_t w : {a, b, c})
                for (int i = 0; i < 3; ++i) {
                    const float x = t.pos[size_t(w) * 3 + i];
                    if (x < t.mn[i]) t.mn[i] = x;
                    if (x > t.mx[i]) t.mx[i] = x;
                }
        }
    }
    if (t.idx.empty()) return fail("no terrain triangles");
    return true;
}

}  // namespace skylevel
