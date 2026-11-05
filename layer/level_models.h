// SkyRT level models reader: every placed mesh of a level, in world space, from the game's own files, so rays see buildings, rocks
// and gates that are off the screen. Placement comes from Objects.level.bin (TGCL, the layout of Stalker 31's tgcl.py), the meshes
// from Data/Meshes/Bin/*.mesh (container 0x20, LZ4 payload, the layout of Stalker 31's skymesh.py). Only LOD 0 positions and indices.
#pragma once
#include <windows.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include "level_terrain.h"   // FindLevelFile / ReadFileAll

namespace skylevel {

struct Models {
    std::vector<float> pos;        // xyz per vertex, world space
    std::vector<uint32_t> idx;     // triangles
    uint32_t placed = 0, skippedShader = 0, skippedHidden = 0, skippedDisabled = 0, skippedBones = 0, missing = 0, broken = 0;
    float mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f};
};

// LZ4 block format
inline bool Lz4Block(const uint8_t* src, size_t n, std::vector<uint8_t>& out, size_t usize) {
    out.clear();
    out.reserve(usize);
    size_t i = 0;
    while (i < n) {
        const uint8_t tok = src[i++];
        size_t ll = tok >> 4;
        if (ll == 15) { uint8_t s; do { if (i >= n) return false; s = src[i++]; ll += s; } while (s == 255); }
        if (i + ll > n) return false;
        out.insert(out.end(), src + i, src + i + ll);
        i += ll;
        if (i >= n) break;
        if (i + 2 > n) return false;
        const size_t off = src[i] | (src[i + 1] << 8);
        i += 2;
        size_t ml = tok & 15;
        if (ml == 15) { uint8_t s; do { if (i >= n) return false; s = src[i++]; ml += s; } while (s == 255); }
        ml += 4;
        if (off == 0 || off > out.size() || out.size() + ml > usize + 64) return false;
        const size_t st = out.size() - off;
        for (size_t k = 0; k < ml; ++k) out.push_back(out[st + k]);
    }
    return out.size() == usize;
}

// LOD 0 of a .mesh: positions (model space) and triangle indices; false for skinned or unreadable meshes
inline bool ReadMeshLod0(const std::vector<uint8_t>& d, const std::wstring& fileName, std::vector<float>& pos, std::vector<uint32_t>& idx, bool& bones) {
    auto has = [&](const wchar_t* s) { return fileName.find(s) != std::wstring::npos; };
    const bool zipPos = has(L"ZipPos"), zipUvs = has(L"ZipUvs"), stripNorm = has(L"StripNorm");
    if (d.size() < 90) return false;
    uint32_t magic; memcpy(&magic, d.data(), 4);
    if (magic != 0x20) return false;
    uint16_t hb; memcpy(&hb, d.data() + 72, 2);
    bones = hb == 1;
    if (bones) return false;
    uint32_t comp, uncomp; memcpy(&comp, d.data() + 82, 4); memcpy(&uncomp, d.data() + 86, 4);
    if (comp == 0 || uncomp == 0 || uncomp > (256u << 20) || 90ull + comp > d.size()) return false;
    std::vector<uint8_t> pl;
    if (!Lz4Block(d.data() + 90, comp, pl, uncomp)) return false;
    const size_t s = 0;
    if (pl.size() < 179) return false;
    auto u32 = [&](size_t o) { uint32_t v; memcpy(&v, pl.data() + o, 4); return v; };
    const uint32_t V = u32(s + 116), I = u32(s + 120), P = u32(s + 128), R = u32(s + 136), P2 = u32(s + 140);
    if (V == 0 || I < 3 || V > (4u << 20) || I > (24u << 20)) return false;
    float aabb2[6]; memcpy(aabb2, pl.data() + s + 28, sizeof aabb2);
    size_t o = s + 179;
    pos.resize((size_t)V * 3);
    auto need = [&](size_t bytes) { return o + bytes <= pl.size(); };
    if (!zipPos) {
        if (!need((size_t)V * 16)) return false;
        for (uint32_t v = 0; v < V; ++v) memcpy(&pos[(size_t)v * 3], pl.data() + o + (size_t)v * 16, 12);
        o += (size_t)V * 16;
    }
    if (!stripNorm) o += (size_t)V * 4;
    if (!zipUvs) o += (size_t)V * 16;
    if (!need((size_t)I * 2)) return false;
    idx.resize(I);
    for (uint32_t k = 0; k < I; ++k) { uint16_t x; memcpy(&x, pl.data() + o + (size_t)k * 2, 2); idx[k] = x; }
    o += (size_t)I * 2;
    o += (size_t)V * 4;                                  // A, B
    if (!(R == 0 && P2 == 0)) o += (size_t)R * 2 + (size_t)P * 4;   // C, D
    o += (size_t)(I / 3) * 4;                            // E
    if (zipPos) {
        if (!need((size_t)V * 4)) return false;
        for (uint32_t v = 0; v < V; ++v) {
            uint32_t q; memcpy(&q, pl.data() + o + (size_t)v * 4, 4);
            const float c[3] = {((q >> 20) & 1023) / 1023.0f, ((q >> 10) & 1023) / 1023.0f, (q & 1023) / 1023.0f};
            for (int a = 0; a < 3; ++a) pos[(size_t)v * 3 + a] = aabb2[a] + c[a] * (aabb2[3 + a] - aabb2[a]);
        }
    }
    for (uint32_t k = 0; k < I; ++k) if (idx[k] >= V) return false;
    idx.resize(I / 3 * 3);
    return true;
}

// Which level meshes block light: solid shaders only (no alpha, fog, light shafts, cloth, flags, foliage, spirits, NoShadows)
inline bool SolidShader(const std::string& s) {
    const bool family = s.rfind("Mesh", 0) == 0 || s.rfind("DarkStone", 0) == 0 || s.find("ShMesh") != std::string::npos;
    if (!family) return false;
    static const char* bad[] = {"Alpha", "NoShadows", "WindCloth", "Flag", "Foliage", "Fading", "Glow", "Cloud", "Fog", "Spirit", "Motion", "Bubble", "Balloon"};
    for (const char* b : bad) if (s.find(b) != std::string::npos) return false;
    return true;
}

// TGCL: types (name, first field, field count), fields (kind, name, size, x), then objects: u32 type, name, field values
struct Tgcl {
    const std::vector<uint8_t>* d = nullptr;
    uint32_t off[5] = {};
    struct Type { std::string name; uint32_t f0, fc; };
    struct Field { uint32_t kind, size, x; std::string name; };
    std::vector<Type> types;
    std::vector<Field> fields;
    bool ok = false;
    std::string Pool(uint32_t o) const {
        const size_t a = off[2] + o;
        size_t e = a;
        while (e < off[3] && e < d->size() && (*d)[e]) ++e;
        return a < d->size() ? std::string((const char*)d->data() + a, e - a) : std::string();
    }
    uint32_t U32(size_t o) const { uint32_t v = 0; if (o + 4 <= d->size()) memcpy(&v, d->data() + o, 4); return v; }
    explicit Tgcl(const std::vector<uint8_t>& data) : d(&data) {
        if (data.size() < 44 || memcmp(data.data(), "TGCL", 4) != 0) return;
        const uint32_t nt = U32(8), nf = U32(12);
        for (int i = 0; i < 5; ++i) off[i] = U32(24 + 4 * i);
        if (off[3] > data.size() || off[2] > off[3] || nt > 100000 || nf > 1000000) return;
        for (uint32_t i = 0; i < nt; ++i) types.push_back({"", U32(off[0] + 12 * i + 4), U32(off[0] + 12 * i + 8)}), types.back().name = Pool(U32(off[0] + 12 * i));
        for (uint32_t i = 0; i < nf; ++i) { const size_t b = off[1] + 16 * (size_t)i; fields.push_back({U32(b), U32(b + 8), U32(b + 12), Pool(U32(b + 4))}); }
        ok = true;
    }
    // skips the values of one type's fields starting at o; with `top`, reports the top-level ones (name, offset, size or string)
    template <class F> bool Walk(uint32_t type, size_t& o, F&& top, int depth) const {
        if (type >= types.size() || depth > 16) return false;
        const Type& t = types[type];
        for (uint32_t k = 0; k < t.fc; ++k) {
            if (t.f0 + k >= fields.size()) return false;
            const Field& f = fields[t.f0 + k];
            if (f.kind == 0) { if (o + f.size > d->size()) return false; if (depth == 0) top(f.name, o, f.size); o += f.size; }
            else if (f.kind == 1) { size_t e = o; while (e < d->size() && (*d)[e]) ++e; if (e >= d->size()) return false; if (depth == 0) top(f.name, o, e - o); o = e + 1; }
            else if (f.kind == 2) { if (o + 4 > d->size()) return false; o += 4; }
            else if (f.kind == 3) {
                if (o + 4 > d->size()) return false;
                const uint32_t c = U32(o); o += 4;
                if (f.x == 0xFFFFFFFFu) { if (o + 4ull * c > d->size()) return false; o += 4ull * c; }
                else for (uint32_t r = 0; r < c; ++r) if (!Walk(f.x, o, top, depth + 1)) return false;   // nested records: top is only called at depth 0
            } else return false;
        }
        return true;
    }
};

// every placed solid LevelMesh of the level, in world space
inline bool LoadModels(const std::wstring& gameRoot, const std::wstring& level, Models& m, std::string* err = nullptr, bool withDisabled = false) {
    auto fail = [&](const char* s) { if (err) *err = s; return false; };
    std::wstring bstPath;
    if (!FindLevelFile(gameRoot, level, bstPath)) return fail("level not found");
    const std::wstring objPath = bstPath.substr(0, bstPath.find_last_of(L"\\/")) + L"\\Objects.level.bin";
    std::vector<uint8_t> d;
    if (!ReadFileAll(objPath, d)) return fail("no Objects.level.bin");
    Tgcl t(d);
    if (!t.ok) return fail("not TGCL");
    // the meshes of all packs, by name
    std::unordered_map<std::wstring, std::wstring> meshes;
    {
        const std::wstring assets = gameRoot + L"\\data\\assets\\";
        WIN32_FIND_DATAW pk;
        HANDLE hp = FindFirstFileW((assets + L"*").c_str(), &pk);
        if (hp != INVALID_HANDLE_VALUE) {
            do {
                if (!(pk.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) || pk.cFileName[0] == L'.') continue;
                const std::wstring dir = assets + pk.cFileName + L"\\Data\\Meshes\\Bin\\";
                WIN32_FIND_DATAW fd;
                HANDLE hf = FindFirstFileW((dir + L"*.mesh").c_str(), &fd);
                if (hf == INVALID_HANDLE_VALUE) continue;
                do {
                    std::wstring n = fd.cFileName;
                    meshes.emplace(n.substr(0, n.size() - 5), dir + n);
                } while (FindNextFileW(hf, &fd));
                FindClose(hf);
            } while (FindNextFileW(hp, &pk));
            FindClose(hp);
        }
    }
    struct Cached { std::vector<float> pos; std::vector<uint32_t> idx; bool ok; };
    std::unordered_map<std::wstring, Cached> cache;
    uint32_t levelMesh = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < t.types.size(); ++i) if (t.types[i].name == "LevelMesh") levelMesh = i;
    size_t o = t.off[3];
    while (o + 4 < d.size()) {
        const uint32_t type = t.U32(o);
        size_t e = o + 4;
        while (e < d.size() && d[e]) ++e;
        o = e + 1;
        std::string res, shader;
        int render = 1, cloud = 0, invisible = 0, enabled = 1;
        float xf[16] = {};
        bool haveXf = false;
        auto top = [&](const std::string& name, size_t at, size_t size) {
            if (type != levelMesh) return;
            if (name == "resourceName") res.assign((const char*)d.data() + at, size);
            else if (name == "shaderName") shader.assign((const char*)d.data() + at, size);
            else if (name == "render" && size >= 1) render = d[at];
            else if (name == "isCloud" && size >= 1) cloud = d[at];
            else if (name == "isInvisibleColliderOnly" && size >= 1) invisible = d[at];
            else if (name == "enabled" && size >= 1) enabled = d[at];
            else if (name == "transform" && size == 64) { memcpy(xf, d.data() + at, 64); haveXf = true; }
        };
        if (!t.Walk(type, o, top, 0)) break;
        if (type != levelMesh || res.empty() || !haveXf) continue;
        if (!render || cloud || invisible) { ++m.skippedHidden; continue; }
        if (!enabled && !withDisabled) { ++m.skippedDisabled; continue; }   // switched on later by the level's scripts: unknown whether it is there now
        if (!SolidShader(shader)) { ++m.skippedShader; continue; }
        const std::wstring wres(res.begin(), res.end());
        auto it = cache.find(wres);
        if (it == cache.end()) {
            Cached c{{}, {}, false};
            auto mf = meshes.find(wres);
            if (mf != meshes.end()) {
                std::vector<uint8_t> md;
                bool bones = false;
                if (ReadFileAll(mf->second, md)) c.ok = ReadMeshLod0(md, mf->first, c.pos, c.idx, bones);
                if (bones) ++m.skippedBones;
                else if (!c.ok) ++m.broken;
            } else ++m.missing;
            it = cache.emplace(wres, std::move(c)).first;
        }
        if (!it->second.ok) continue;
        bool finite = true;
        for (float v : xf) if (!(v == v) || v > 1e7f || v < -1e7f) finite = false;
        if (!finite) continue;
        const uint32_t base = (uint32_t)(m.pos.size() / 3);
        const std::vector<float>& p = it->second.pos;
        for (size_t v = 0; v + 2 < p.size(); v += 3) {   // column-major matrix: world = M * (x, y, z, 1)
            const float x = p[v], y = p[v + 1], z = p[v + 2];
            const float w[3] = {xf[0] * x + xf[4] * y + xf[8] * z + xf[12], xf[1] * x + xf[5] * y + xf[9] * z + xf[13], xf[2] * x + xf[6] * y + xf[10] * z + xf[14]};
            for (int a = 0; a < 3; ++a) { m.pos.push_back(w[a]); if (w[a] < m.mn[a]) m.mn[a] = w[a]; if (w[a] > m.mx[a]) m.mx[a] = w[a]; }
        }
        for (uint32_t k : it->second.idx) m.idx.push_back(base + k);
        ++m.placed;
        if (m.idx.size() > 3u * 4000000u) return fail("more than 4 million triangles");
    }
    if (m.idx.empty()) return fail("no solid models");
    return true;
}

}  // namespace skylevel
