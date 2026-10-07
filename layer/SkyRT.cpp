// SkyRT.cpp - implicit Vulkan layer, step 1.
//   * console + log file (flushed every line) + crash log (exception, module+offset, stack)
//   * enables ray_query / acceleration_structure on the game's VkDevice (with automatic fallback)
//   * patches vertex/index/indirect buffers and memory so they get a device address (needed for BLAS)
//   * counts draw calls per frame
// Build: build.bat (x64 Native Tools Command Prompt, Vulkan SDK installed).

#define _CRT_SECURE_NO_WARNINGS
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <io.h>
#include <stdio.h>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rt_spv.h"    // SPIR-V of rt.comp (generated): trace pass
#include "rtfx_spv.h"  // SPIR-V of rt_fx.comp (generated): denoise + composite pass
#include "rtlight_spv.h"  // SPIR-V of rt_light.comp (generated): emissive light source search (step 21)

// ============================================================ config
static const uint32_t kStatsEveryNFrames = 300;   // ~5 sec at 60 fps
static const int      kMaxBufferLogLines = 400;
static const uint32_t kApi12 = (1u << 22) | (2u << 12);

static bool g_noPaint = false;                   // SKYRT_NO_PAINT=1: do not inject the test compute pass
static bool g_noImgPatch = false;                // SKYRT_NO_IMGPATCH=1: do not add STORAGE/SAMPLED to the main attachments
static bool g_logOnly = false;                    // SKYRT_LOGONLY=1: change nothing, only log
static double g_sunAz = 30.0, g_sunEl = 50.0;    // sun direction (degrees), adjustable with RightCtrl + arrows
static float g_strength = 0.5f;                  // how much a shadow ray darkens the pixel
static bool g_enabled = true;                    // Ctrl+Home: ray tracing on / off (original game picture)
static uint32_t g_view = 0;                      // Ctrl+End: cycles through the views below
static const char* const kViewNames[8] = {"full (shadows + AO + bounce light)", "shadows only", "AO only", "bounce light only",
                                          "DEBUG: AO map", "DEBUG: bounce light map", "DEBUG: world grid", "DEBUG: normals"};
static uint32_t g_mode = 0;                      // 0 shadows, 1 world grid, 2 normals, 3 off
static bool g_shadows = true, g_aoOn = true;      // effects
static float g_aoStrength = 0.7f, g_aoRadius = 1.5f;
static int g_aoRays = 4;
static bool g_giOn = true;
// step 11: device memory telemetry (what the game allocates, per heap)
static std::atomic<uint64_t> g_heapLive[16], g_heapPeak[16], g_allocFails{0}, g_allocBig{0};
static uint64_t g_heapSize[16] = {};
static uint32_t g_heapCount = 0;
static bool g_reflOn = false;                         // step 9: ray traced water
static float g_reflStrength = 0.5f, g_glint = 1.0f;
static int g_dumpShaders = 1;    // step 15: write every SPIR-V module to SkyRT_shaders\<fnv>.spv (cfg dumpshaders=0/1)
static int g_instGeo = 1;         // step 18: instanced props (few instances, known layout) -> baked into the per-frame dynamic BLAS with per-instance transforms (cfg instgeo=0/1)
static float g_instScale = 1.0f;  // step 18: multiplier on the per-instance scale (cfg instscale)
static const uint64_t kInstVs[] = {0xd51780d883c920d3ull, 0x98266faa42756ec4ull};   // vertex shaders whose instance layout is decoded: float3 position @0, float scale @12
static const uint32_t kXfCap = 2048;
static const uint32_t kInstMaxPerDraw = 48, kInstMinIdx = 90, kPropGeomMax = 1000;
static const uint64_t kPropTrisMax = 100000;
static int g_taa = 1;             // step 20: temporal accumulation of shadows / AO / GI (cfg taa=0/1)
static int g_lightOn = 1;          // step 21: light from fire / candles / lamps (bright pixels of the frame become point lights), cfg light=0/1
static float g_lightStrength = 1.0f, g_lightRange = 12.0f, g_lightThr = 2.0f;   // cfg lightstrength, lightrange (world units), lightthr (HDR brightness that counts as emissive)
static int g_lightRays = 2, g_lightDebug = 0;   // cfg lightrays (shadow rays per pixel), lightdebug (paint the emissive pixels magenta)
static float g_taaN = 12.0f;      // step 20: maximum accumulated samples (cfg taan); lower = less ghosting, more noise
static int g_dynGeo = 1;          // step 16: geometry in CPU-written (host visible) vertex buffers = skinned characters -> own BLAS rebuilt every frame (cfg dyngeo=0/1)
static int g_aniso = 16;          // step 13: anisotropic filtering forced on every linear sampler (cfg: aniso=0/2/4/8/16)
static float g_lodBias = 0.0f;    // cfg: lodbias (negative = sharper textures, e.g. -0.5)
static float g_giStrength = 0.35f, g_giRange = 30.0f, g_sunSize = 2.5f;  // sun size in degrees (soft shadow penumbra)
static int g_shRays = 2;
static bool g_autoSun = true;                     // take the sun direction from the game UBO (offset 560)
static bool g_flipY = false;                     // SKYRT_FLIPY=1
static std::atomic<const char*> g_stage{"startup"};  // breadcrumb: last layer entry point

// ============================================================ logging
static std::mutex g_logMtx;
static FILE* g_file = nullptr;
static bool g_console = false;
static bool g_ready = false;
static std::atomic<bool> g_noLock{false};
static std::once_flag g_initOnce;
static char g_logPath[MAX_PATH] = {};
static LPTOP_LEVEL_EXCEPTION_FILTER g_prevUef = nullptr;

static bool EnvIs1(const char* name) {
    char b[8] = {};
    DWORD n = GetEnvironmentVariableA(name, b, sizeof b);
    return n > 0 && n < sizeof b && b[0] == '1';
}

static void ModuleDir(char* out, size_t n) {
    HMODULE hm = nullptr;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCSTR)&ModuleDir, &hm);
    GetModuleFileNameA(hm, out, (DWORD)n);
    char* s = strrchr(out, '\\');
    if (s) *s = 0;
}

static void Logf(const char* fmt, ...) {
    if (!g_ready) return;
    char msg[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    SYSTEMTIME st;
    GetLocalTime(&st);
    char line[2200];
    snprintf(line, sizeof line, "[%02d:%02d:%02d.%03d][t%05lu] %s\n", st.wHour, st.wMinute, st.wSecond,
             st.wMilliseconds, GetCurrentThreadId(), msg);
    std::unique_lock<std::mutex> lk(g_logMtx, std::defer_lock);
    if (!g_noLock.load()) lk.lock();
    if (g_file) { fputs(line, g_file); fflush(g_file); }
    if (g_console) { fputs(line, stdout); fflush(stdout); }
    OutputDebugStringA(line);
}

static void FlushToDisk() {
    if (g_file) { fflush(g_file); _commit(_fileno(g_file)); }
}

// ------------------------------------------------------------ crash log
static void DescribeAddr(const void* addr, char* out, size_t n) {
    HMODULE hm = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR)addr, &hm) && hm) {
        char path[MAX_PATH] = {};
        GetModuleFileNameA(hm, path, MAX_PATH);
        const char* base = strrchr(path, '\\');
        base = base ? base + 1 : path;
        snprintf(out, n, "%s+0x%llx", base, (unsigned long long)((uintptr_t)addr - (uintptr_t)hm));
    } else {
        snprintf(out, n, "0x%llx", (unsigned long long)(uintptr_t)addr);
    }
}

static bool IsFatalCode(DWORD c) {
    switch (c) {
        case EXCEPTION_ACCESS_VIOLATION:
        case EXCEPTION_ILLEGAL_INSTRUCTION:
        case EXCEPTION_STACK_OVERFLOW:
        case EXCEPTION_INT_DIVIDE_BY_ZERO:
        case EXCEPTION_PRIV_INSTRUCTION:
        case EXCEPTION_IN_PAGE_ERROR:
        case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:
        case 0xC0000374:  // heap corruption
        case 0xC0000409:  // stack buffer overrun / fail fast
            return true;
    }
    return false;
}

static void LogException(const char* tag, EXCEPTION_POINTERS* ep) {
    bool prevNoLock = g_noLock.exchange(true);  // the crashed thread may hold the log mutex
    const EXCEPTION_RECORD* er = ep->ExceptionRecord;
    char where[300];
    DescribeAddr(er->ExceptionAddress, where, sizeof where);
    Logf("!!! %s: code=0x%08lX at %s (thread %lu), last layer stage: %s", tag, er->ExceptionCode, where,
         GetCurrentThreadId(), g_stage.load());
    if (er->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && er->NumberParameters >= 2) {
        const char* kind = er->ExceptionInformation[0] == 0 ? "read" : er->ExceptionInformation[0] == 1 ? "write" : "execute";
        Logf("!!!   access violation: %s of address 0x%llx", kind, (unsigned long long)er->ExceptionInformation[1]);
    }
    void* frames[40];
    USHORT n = CaptureStackBackTrace(0, 40, frames, nullptr);
    for (USHORT i = 0; i < n; ++i) {
        char b[300];
        DescribeAddr(frames[i], b, sizeof b);
        Logf("!!!   #%02u %s", (unsigned)i, b);
    }
    FlushToDisk();
    g_noLock = prevNoLock;
}

static LONG CALLBACK VehHandler(EXCEPTION_POINTERS* ep) {
    static std::atomic<int> count{0};
    if (IsFatalCode(ep->ExceptionRecord->ExceptionCode) && count.fetch_add(1) < 25)
        LogException("first-chance exception (the game may handle it itself)", ep);
    return EXCEPTION_CONTINUE_SEARCH;
}

static LONG WINAPI UefHandler(EXCEPTION_POINTERS* ep) {
    LogException("UNHANDLED EXCEPTION - the game is crashing", ep);
    return g_prevUef ? g_prevUef(ep) : EXCEPTION_CONTINUE_SEARCH;
}

static void LogInit() {
    std::call_once(g_initOnce, [] {
        g_logOnly = EnvIs1("SKYRT_LOGONLY");
        g_noImgPatch = EnvIs1("SKYRT_NO_IMGPATCH");
        g_noPaint = EnvIs1("SKYRT_NO_PAINT");
        g_flipY = EnvIs1("SKYRT_FLIPY");
        if (const char* e = getenv("SKYRT_SUN")) { double a = 0, b = 0; if (sscanf(e, "%lf,%lf", &a, &b) == 2) { g_sunAz = a; g_sunEl = b; } }
        if (const char* e = getenv("SKYRT_VIEW")) g_view = (uint32_t)atoi(e) % 8u;
        if (const char* e = getenv("SKYRT_STRENGTH")) { float v = (float)atof(e); if (v >= 0.f && v <= 1.f) g_strength = v; }

        // console: the close button is removed (closing it would kill the game),
        // quick-edit is off (selecting text in the console would freeze the game)
        if (AllocConsole()) {
            FILE* fp = nullptr;
            freopen_s(&fp, "CONOUT$", "w", stdout);
            freopen_s(&fp, "CONOUT$", "w", stderr);
            SetConsoleTitleA("SkyRT layer log");
            HWND hw = GetConsoleWindow();
            if (hw) {
                HMENU m = GetSystemMenu(hw, FALSE);
                if (m) DeleteMenu(m, SC_CLOSE, MF_BYCOMMAND);
            }
            HANDLE hin = GetStdHandle(STD_INPUT_HANDLE);
            DWORD mode = 0;
            if (hin && GetConsoleMode(hin, &mode)) {
                mode &= ~ENABLE_QUICK_EDIT_MODE;
                mode |= ENABLE_EXTENDED_FLAGS;
                SetConsoleMode(hin, mode);
            }
            g_console = true;
        }

        // log file next to the dll (fallback: %TEMP%)
        char dir[MAX_PATH] = {};
        ModuleDir(dir, sizeof dir);
        SYSTEMTIME st;
        GetLocalTime(&st);
        snprintf(g_logPath, sizeof g_logPath, "%s\\SkyRT_%04d%02d%02d_%02d%02d%02d_pid%lu.log", dir, st.wYear,
                 st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond, GetCurrentProcessId());
        if (fopen_s(&g_file, g_logPath, "w") != 0 || !g_file) {
            char tmp[MAX_PATH] = {};
            GetTempPathA(MAX_PATH, tmp);
            snprintf(g_logPath, sizeof g_logPath, "%sSkyRT_%04d%02d%02d_%02d%02d%02d_pid%lu.log", tmp, st.wYear,
                     st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond, GetCurrentProcessId());
            fopen_s(&g_file, g_logPath, "w");
        }
        g_ready = true;

        AddVectoredExceptionHandler(1, VehHandler);
        g_prevUef = SetUnhandledExceptionFilter(UefHandler);

        char exe[MAX_PATH] = {};
        GetModuleFileNameA(nullptr, exe, MAX_PATH);
        Logf("=== SkyRT layer loaded === pid=%lu exe=%s", GetCurrentProcessId(), exe);
        Logf("log file: %s", g_logPath);
        Logf("mode: %s", g_logOnly ? "LOG ONLY (game calls are not modified)" : "FULL (RT patching enabled)");
        Logf("if this log has no 'clean shutdown' line at the end, the game crashed or was killed");
    });
}

static void ApiStr(uint32_t v, char* o, size_t n) {
    snprintf(o, n, "%u.%u.%u", (v >> 22) & 0x7Fu, (v >> 12) & 0x3FFu, v & 0xFFFu);
}

// ============================================================ layer state
// persistent geometry cache: world meshes seen in the main pass in recent frames (the game culls to the view,
// shadow rays need occluders that are off screen too)
struct GeoKey {
    VkBuffer vb, ib;
    uint64_t vbase, ibase;
    uint32_t count, stride;
    bool operator==(const GeoKey& o) const {
        return vb == o.vb && ib == o.ib && vbase == o.vbase && ibase == o.ibase && count == o.count && stride == o.stride;
    }
};
struct GeoKeyHash {
    size_t operator()(const GeoKey& k) const {
        uint64_t h = 1469598103934665603ull;
        const uint64_t v[5] = {(uint64_t)(uintptr_t)k.vb, (uint64_t)(uintptr_t)k.ib, k.vbase, k.ibase,
                               ((uint64_t)k.count << 32) | k.stride};
        for (uint64_t x : v) { h ^= x + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); h *= 1099511628211ull; }
        return (size_t)h;
    }
};
struct GeoEnt {
    VkAccelerationStructureGeometryKHR g;
    VkAccelerationStructureBuildRangeInfoKHR r;
    uint64_t lastSeen;
    VkDeviceMemory vm, im;  // memory blocks behind the vertex / index buffers
};
static const uint64_t kMaxCacheTris = 700000;
static const uint64_t kDynTrisMax = 200000;   // step 16: more than this in CPU-written buffers is not 'characters' any more
static const size_t kMaxCacheGeoms = 30000;

struct InstanceData {
    VkInstance instance;
    PFN_vkGetInstanceProcAddr gipa;
    PFN_vkDestroyInstance DestroyInstance;
    PFN_vkCreateDevice CreateDevice;
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties;
    PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties;
    PFN_vkGetPhysicalDeviceFeatures2 GetPhysicalDeviceFeatures2;
    PFN_vkGetPhysicalDeviceProperties2 GetPhysicalDeviceProperties2;
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties;
};

struct DeviceData {
    VkDevice device;
    InstanceData* inst;
    VkPhysicalDevice phys;
    PFN_vkGetDeviceProcAddr gdpa;
    PFN_vkDestroyDevice DestroyDevice;
    PFN_vkCreateBuffer CreateBuffer;
    PFN_vkAllocateMemory AllocateMemory;
    PFN_vkQueuePresentKHR QueuePresentKHR;
    PFN_vkQueueSubmit QueueSubmit;
    PFN_vkCmdDrawIndexed CmdDrawIndexed;
    PFN_vkCmdDraw CmdDraw;
    PFN_vkCmdDrawIndexedIndirect CmdDrawIndexedIndirect;
    PFN_vkCreateSampler CreateSampler;
    bool anisoOk = false;
    float maxAniso = 1.0f;
    PFN_vkCmdDrawIndirect CmdDrawIndirect;
    PFN_vkCmdDrawIndirectCount CmdDrawIndirectCount;
    PFN_vkCmdDrawIndexedIndirectCount CmdDrawIndexedIndirectCount;
    PFN_vkCmdExecuteCommands CmdExecuteCommands;
    PFN_vkBindBufferMemory BindBufferMemory;
    PFN_vkBindBufferMemory2 BindBufferMemory2;
    PFN_vkMapMemory MapMemory;
    PFN_vkUnmapMemory UnmapMemory;
    PFN_vkFreeMemory FreeMemory;
    PFN_vkDestroyBuffer DestroyBuffer;
    PFN_vkGetBufferDeviceAddress GetBufferDeviceAddress;
    bool rtEnabled;

    // registry of interesting buffers / memory blocks (filled by the bind / map hooks)
    struct BufInfo { VkDeviceSize size; VkBufferUsageFlags usage; VkDeviceMemory mem; VkDeviceSize memOff; };
    struct MemInfo { VkDeviceSize size; char* mapped; VkDeviceSize mapOffset; VkDeviceSize mapSize; uint32_t heap = 0xFFFFFFFFu; bool hostVisible = false; };
    std::mutex regMtx;
    std::unordered_map<VkBuffer, BufInfo> bufs;
    std::unordered_map<VkDeviceMemory, MemInfo> mems;
    std::atomic<uint64_t> bindCalls1{0}, bindCalls2{0};

    // frame capture: armed from vkQueuePresentKHR, records one frame of draws (see ArmCapture / FinishCapture)
    PFN_vkBeginCommandBuffer BeginCommandBuffer;
    PFN_vkCmdBindVertexBuffers CmdBindVertexBuffers;
    PFN_vkCmdBindIndexBuffer CmdBindIndexBuffer;
    PFN_vkCreateGraphicsPipelines CreateGraphicsPipelines;
    PFN_vkDestroyPipeline DestroyPipeline;
    PFN_vkCmdBindPipeline CmdBindPipeline;
    PFN_vkCreateSwapchainKHR CreateSwapchainKHR;
    // static vertex layout of every graphics pipeline (guarded by regMtx)
    struct PipeInfo {
        struct Bind { uint32_t binding, stride, rate; };
        struct Attr { uint32_t loc, binding, format, offset; };
        std::vector<Bind> binds;
        std::vector<Attr> attrs;
        bool opaque = false;     // depth write on and no blending (sky dome / particles / water are excluded)
        bool instanced = false;  // has an instance-rate binding (object-space geometry, skipped)
        uint64_t vsHash = 0, fsHash = 0;      // FNV-1a of the SPIR-V of the vertex / fragment shader (stable between runs)
        bool depthTest = false, depthWrite = false, blend = false;
        uint32_t srcColor = 0, dstColor = 0;  // blend factors of attachment 0
    };
    std::unordered_map<VkShaderModule, uint64_t> smHash;   // regMtx
    std::unordered_map<VkPipeline, PipeInfo> pipes;
    struct CmdState { VkBuffer vb[8]; VkDeviceSize vbOff[8]; VkBuffer ib; VkDeviceSize ibOff; VkIndexType ibType; VkPipeline pipe; };
    struct DrawRec {
        CmdState st;
        bool indirect;
        VkBuffer indBuf;
        VkDeviceSize indOff;
        uint32_t drawCount, stride;
        VkDrawIndexedIndirectCommand c;  // direct draws only
        uint64_t recHash;                // hash of the indirect args read at record time
        bool recRead;                    // were they readable at record time
    };
    struct Resolved { CmdState st; VkDrawIndexedIndirectCommand c; bool fromIndirect; };
    struct Group { VkBuffer indBuf; VkDeviceSize indOff; uint32_t drawCount, stride; CmdState st; bool readable; };
    std::atomic<bool> capturing{false};
    std::mutex cmdMtx;
    std::unordered_map<VkCommandBuffer, CmdState> cmdState;
    std::unordered_map<VkCommandBuffer, std::vector<DrawRec>> cmdDraws;
    std::vector<Resolved> resolved;  // guarded by cmdMtx
    std::vector<Group> groups;       // guarded by cmdMtx
    uint64_t unreadable = 0;         // guarded by cmdMtx

    // step 3: image / render pass logging, queue families, BLAS smoke test
    PFN_vkCreateImage CreateImage;
    PFN_vkCreateRenderPass CreateRenderPass;
    PFN_vkCreateRenderPass2 CreateRenderPass2;
    PFN_vkCmdBeginRenderPass CmdBeginRenderPass;
    PFN_vkCmdBeginRenderPass2 CmdBeginRenderPass2;
    PFN_vkGetDeviceQueue GetDeviceQueue;
    PFN_vkGetDeviceQueue2 GetDeviceQueue2;
    PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements;
    PFN_vkCreateCommandPool CreateCommandPool;
    PFN_vkDestroyCommandPool DestroyCommandPool;
    PFN_vkAllocateCommandBuffers AllocateCommandBuffers;
    PFN_vkEndCommandBuffer EndCommandBuffer;
    PFN_vkCreateFence CreateFence;
    PFN_vkDestroyFence DestroyFence;
    PFN_vkWaitForFences WaitForFences;
    PFN_vkGetAccelerationStructureBuildSizesKHR GetAccelerationStructureBuildSizesKHR;
    PFN_vkCreateAccelerationStructureKHR CreateAccelerationStructureKHR;
    PFN_vkDestroyAccelerationStructureKHR DestroyAccelerationStructureKHR;
    PFN_vkCmdBuildAccelerationStructuresKHR CmdBuildAccelerationStructuresKHR;
    PFN_vkSetDeviceLoaderData setLoaderData;
    std::unordered_map<VkQueue, uint32_t> queueFamily;  // guarded by regMtx
    std::atomic<uint64_t> rpBegin1{0}, rpBegin2{0}, imgLogged{0}, rpLogged{0};
    std::atomic<bool> blasDone{false};
    std::atomic<VkQueue> presentQueue{VK_NULL_HANDLE};

    // step 4: attachment images, framebuffers, render passes of the captured frame
    PFN_vkCreateImageView CreateImageView;
    PFN_vkCreateFramebuffer CreateFramebuffer;
    PFN_vkCmdEndRenderPass CmdEndRenderPass;
    struct ImgInfo { VkFormat format; uint32_t w, h; VkImageUsageFlags usage, patchedUsage; };
    struct ViewInfo { VkImage image; };
    struct FbInfo { VkRenderPass rp; uint32_t w, h; std::vector<VkImageView> views; };
    struct AttInfo { uint32_t format, loadOp, storeOp, initialLayout, finalLayout; };
    std::unordered_map<VkImage, ImgInfo> imgs;                    // big color/depth attachment images only (regMtx)
    std::unordered_map<VkImageView, ViewInfo> views;              // views of the images above (regMtx)
    std::unordered_map<VkFramebuffer, FbInfo> fbs;                // framebuffers that use them (regMtx)
    std::unordered_map<VkRenderPass, std::vector<AttInfo>> rps;   // attachment ops/layouts of every render pass (regMtx)
    struct PassEv { VkRenderPass rp; VkFramebuffer fb; uint32_t w, h; VkCommandBuffer cb; size_t drawStart, drawEnd; bool open; };
    std::vector<PassEv> passes;                                   // guarded by cmdMtx
    std::atomic<uint64_t> imgPatched{0}, imgPatchFail{0};

    // step 5: compute pass injected after the main render pass
    PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout;
    PFN_vkCreatePipelineLayout CreatePipelineLayout;
    PFN_vkCreateShaderModule CreateShaderModule;
    PFN_vkDestroyShaderModule DestroyShaderModule;
    PFN_vkCreateComputePipelines CreateComputePipelines;
    PFN_vkCreateDescriptorPool CreateDescriptorPool;
    PFN_vkAllocateDescriptorSets AllocateDescriptorSets;
    PFN_vkUpdateDescriptorSets UpdateDescriptorSets;
    PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets;
    PFN_vkCmdPushConstants CmdPushConstants;
    PFN_vkCmdDispatch CmdDispatch;
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier;
    PFN_vkCmdFillBuffer CmdFillBuffer;
    PFN_vkCmdCopyBuffer CmdCopyBuffer;
    PFN_vkDestroyImageView DestroyImageView;
    PFN_vkDestroyImage DestroyImage;
    struct MainCb { VkImage color, depth; uint32_t w, h; };
    std::mutex paintMtx;
    std::unordered_map<VkCommandBuffer, MainCb> mainCb;   // command buffers currently inside the main render pass
    std::unordered_map<VkImage, VkImageView> ownViews;    // our own views of the game's images (key: image)
    VkDescriptorSetLayout paintDsl = VK_NULL_HANDLE;
    VkPipelineLayout paintPl = VK_NULL_HANDLE;
    VkPipeline paintPipe = VK_NULL_HANDLE;
    VkDescriptorPool paintPool = VK_NULL_HANDLE;
    VkDescriptorSet paintSets[16] = {};
    bool paintTried = false, paintOk = false;             // guarded by paintMtx
    std::atomic<uint32_t> paintRing{0};
    std::atomic<uint64_t> paintInjected{0};
    // global frame UBO discovery (descriptor with range 1584 = the struct the vertex shader reads)
    struct UboCand { VkBuffer buf; VkDeviceSize off, range; uint32_t type; };
    std::mutex uboMtx;
    std::vector<UboCand> uboCands;                        // guarded by uboMtx
    std::set<uint32_t> dynOffsets;                        // dynamic offsets seen while capturing (uboMtx)
    std::atomic<bool> uboScanDone{false};
    std::atomic<uint64_t> uboWrites{0}, dynBinds{0};
    // indirect args: read at record time vs at submit time
    std::atomic<uint64_t> recChecked{0}, recDiffer{0}, recUnreadable{0};

    // step 6: per-frame BLAS/TLAS built inside the game's command buffer + ray_query shadow pass
    PFN_vkGetAccelerationStructureDeviceAddressKHR GetAccelerationStructureDeviceAddressKHR;
    struct FrameGeo {
        std::vector<VkAccelerationStructureGeometryKHR> geoms;
        std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
        std::vector<GeoKey> keys;
        std::vector<std::pair<VkDeviceMemory, VkDeviceMemory>> mems;
        uint64_t tris = 0, draws = 0, skipPipe = 0, skipRange = 0, skipOther = 0, unreadable = 0;
        std::vector<VkAccelerationStructureGeometryKHR> dynGeoms;       // step 16: CPU-written vertex buffers (characters)
        std::vector<VkAccelerationStructureBuildRangeInfoKHR> dynRanges;
        std::vector<GeoKey> dynKeys;
        std::vector<std::pair<VkDeviceMemory, VkDeviceMemory>> dynMems;
        uint64_t dynTris = 0;
        std::vector<VkTransformMatrixKHR> dynXf;   // step 18: per dyn geometry transform (identity for characters)
        std::vector<uint8_t> dynProp;              // step 18: 1 = instanced prop (transform used)
        uint64_t propTris = 0;
        uint32_t propGeoms = 0;
    };
    struct DynSlot {
        VkBuffer blasBuf = VK_NULL_HANDLE, tlasBuf = VK_NULL_HANDLE, scratchBuf = VK_NULL_HANDLE, instBuf = VK_NULL_HANDLE;
        VkDeviceMemory blasMem = VK_NULL_HANDLE, tlasMem = VK_NULL_HANDLE, scratchMem = VK_NULL_HANDLE, instMem = VK_NULL_HANDLE;
        VkDeviceAddress scratchAddr = 0, instAddr = 0, blasAddr = 0;
        VkAccelerationStructureKHR blas = VK_NULL_HANDLE, tlas = VK_NULL_HANDLE;
        void* instPtr = nullptr;
        VkBuffer xfBuf = VK_NULL_HANDLE;
        VkDeviceMemory xfMem = VK_NULL_HANDLE;
        VkDeviceAddress xfAddr = 0;
        void* xfPtr = nullptr;
        uint64_t lastUse = 0;
        bool ok = false;
    };
    DynSlot dyn[8];
    bool dynTried = false, dynOk = false;
    std::atomic<uint64_t> dynFrames{0}, dynGeomsSum{0}, dynTrisSum{0}, dynSkipped{0}, dynClassified{0};
    std::mutex geoMtx;
    std::unordered_map<VkCommandBuffer, FrameGeo> frameGeo;          // geoMtx
    std::atomic<int> mainCount{0};
    bool track = false;                                               // track vertex/index/pipeline binds of every cb
    struct RtSlot {
        VkBuffer blasBuf = VK_NULL_HANDLE, tlasBuf = VK_NULL_HANDLE, scratchBuf = VK_NULL_HANDLE, instBuf = VK_NULL_HANDLE;
        VkDeviceMemory blasMem = VK_NULL_HANDLE, tlasMem = VK_NULL_HANDLE, scratchMem = VK_NULL_HANDLE, instMem = VK_NULL_HANDLE;
        VkDeviceAddress scratchAddr = 0, instAddr = 0, blasAddr = 0;
        VkAccelerationStructureKHR blas = VK_NULL_HANDLE, tlas = VK_NULL_HANDLE;
        VkDeviceSize blasCap = 0, tlasCap = 0, scratchCap = 0;
        uint64_t lastUse = 0;  // frame number of the last trace pass that read this slot
        bool ok = false;
    };
    RtSlot rtSlots[8];
    // step 8: trace + composite passes, own result images, per-frame parameter buffers
    PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements;
    PFN_vkBindImageMemory BindImageMemory;
    VkPipeline fxPipe = VK_NULL_HANDLE;
    VkPipeline lightPipe = VK_NULL_HANDLE;           // step 21: emissive light search
    VkBuffer lightBuf = VK_NULL_HANDLE, statBuf = VK_NULL_HANDLE;   // device-local light list / small host-visible copy of its header for the log
    VkDeviceMemory lightMem = VK_NULL_HANDLE, statMem = VK_NULL_HANDLE;
    char* statPtr = nullptr;
    VkImage giImg = VK_NULL_HANDLE, shImg = VK_NULL_HANDLE;
    VkDeviceMemory giMem = VK_NULL_HANDLE, shMem = VK_NULL_HANDLE;
    VkImageView giView = VK_NULL_HANDLE, shView = VK_NULL_HANDLE;
    uint32_t fxW = 0, fxH = 0;
    VkImage hImg[4] = {};            // step 20: history A0,A1 (rgba16f) and B0,B1 (rgba32f), ping-pong
    VkDeviceMemory hMem[4] = {};
    VkImageView hView[4] = {};
    uint32_t histCur = 0;
    bool histValid = false;
    float prevVp[16] = {};
    VkBuffer parBuf[16] = {};
    VkDeviceMemory parMem[16] = {};
    char* parPtr[16] = {};
    std::mutex cacheMtx;
    std::unordered_map<GeoKey, GeoEnt, GeoKeyHash> geoCache;          // cacheMtx
    std::unordered_set<VkBuffer> cachedBufs;                          // cacheMtx
    std::unordered_set<VkDeviceMemory> buildMems;                     // cacheMtx: memory blocks an AS build may read
    PFN_vkDeviceWaitIdle DeviceWaitIdle;
    struct Deferred { VkBuffer buf; VkDeviceMemory mem; uint64_t frame; };  // cacheMtx: game objects whose destruction is held back
    std::vector<Deferred> deferred;
    uint64_t cacheTris = 0, lastEvict = 0;                            // cacheMtx
    bool cacheDirty = false;                                          // cacheMtx: new geometry appeared (rebuild is throttled)
    bool cacheForce = false;                                          // cacheMtx: geometry was removed (rebuild at once, no dangling addresses)
    uint64_t lastRebuildFrame = 0;                                    // cacheMtx
    uint64_t lastRebuildTris = 0;                                     // cacheMtx: cache size at the last BLAS rebuild
    std::atomic<uint64_t> lastRebuildFrameA{0}, hitches{0};
    std::atomic<uint64_t> rtDeferred{0};
    std::atomic<uint64_t> usInject{0}, usInjectMax{0}, usDraw{0}, injectN{0}, spikes{0};
    int rtCur = -1;                                                   // cacheMtx: slot holding the newest complete AS
    std::atomic<uint64_t> rtRebuilds{0}, rtEvicted{0};
    bool rtSlotsTried = false, rtSlotsOk = false;                     // paintMtx
    VkDeviceSize asScratchAlign = 256;
    std::atomic<uint32_t> rtRing{0};
    std::atomic<uint64_t> rtBuilt{0}, rtSkipped{0}, rtLastGeoms{0}, rtLastTris{0}, rtLastAsKB{0}, rtNoCam{0};
    // descriptor set -> frame UBO it contains, and the UBO bound inside the main pass of each command buffer
    struct UboRef { VkBuffer buf; VkDeviceSize off; };
    std::unordered_map<VkDescriptorSet, UboRef> setUbo;               // uboMtx
    std::unordered_map<VkCommandBuffer, UboRef> mainUbo;              // uboMtx

    std::atomic<uint64_t> frames{0}, drawIndexed{0}, indirectCalls{0}, indirectSubdraws{0}, submits{0};
    std::atomic<uint64_t> vtxBuf{0}, idxBuf{0}, indBuf{0}, bufPatched{0}, bufPatchFail{0};
    std::atomic<uint64_t> memPatched{0}, memPatchFail{0};
    uint64_t lastStatFrame = 0;
};

static std::mutex g_mapMtx;
static std::unordered_map<void*, InstanceData*> g_instances;
static std::unordered_map<void*, DeviceData*> g_devices;

// dispatch key = first pointer inside any dispatchable handle (shared by instance/physdev and device/queue/cmdbuf)
template <class T> static inline void* Key(T handle) { return *(void**)handle; }

static InstanceData* GetInst(void* key) {
    std::lock_guard<std::mutex> lk(g_mapMtx);
    auto it = g_instances.find(key);
    return it == g_instances.end() ? nullptr : it->second;
}
static DeviceData* GetDev(void* key) {
    std::lock_guard<std::mutex> lk(g_mapMtx);
    auto it = g_devices.find(key);
    return it == g_devices.end() ? nullptr : it->second;
}

// ============================================================ instance
static VKAPI_ATTR VkResult VKAPI_CALL SkyRT_CreateInstance(const VkInstanceCreateInfo* pCreateInfo,
                                                           const VkAllocationCallbacks* pAllocator,
                                                           VkInstance* pInstance) {
    LogInit();
    g_stage = "vkCreateInstance";

    auto* chain = (VkLayerInstanceCreateInfo*)pCreateInfo->pNext;
    while (chain && !(chain->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
                      chain->function == VK_LAYER_LINK_INFO))
        chain = (VkLayerInstanceCreateInfo*)chain->pNext;
    if (!chain) { Logf("ERROR: no loader link info in vkCreateInstance chain"); return VK_ERROR_INITIALIZATION_FAILED; }

    PFN_vkGetInstanceProcAddr gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;
    auto* savedLink = chain->u.pLayerInfo;  // to restore the chain before a retry

    auto createInstance = (PFN_vkCreateInstance)gipa(VK_NULL_HANDLE, "vkCreateInstance");
    if (!createInstance) return VK_ERROR_INITIALIZATION_FAILED;

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    if (pCreateInfo->pApplicationInfo) app = *pCreateInfo->pApplicationInfo;
    char apiS[32];
    ApiStr(app.apiVersion, apiS, sizeof apiS);
    Logf("game instance: app='%s' engine='%s' apiVersion=%s", app.pApplicationName ? app.pApplicationName : "-",
         app.pEngineName ? app.pEngineName : "-", apiS);
    for (uint32_t i = 0; i < pCreateInfo->enabledExtensionCount; ++i)
        Logf("  instance ext: %s", pCreateInfo->ppEnabledExtensionNames[i]);
    for (uint32_t i = 0; i < pCreateInfo->enabledLayerCount; ++i)
        Logf("  instance layer: %s", pCreateInfo->ppEnabledLayerNames[i]);

    // Raise apiVersion to 1.2: core bufferDeviceAddress + ray_query need it
    bool bumped = false;
    VkInstanceCreateInfo ci = *pCreateInfo;
    if (!g_logOnly && (app.apiVersion & 0x1FFFFFFFu) < kApi12) {
        app.apiVersion = kApi12;
        bumped = true;
        Logf("raising instance apiVersion to 1.2.0");
    }
    ci.pApplicationInfo = &app;

    VkResult r = createInstance(&ci, pAllocator, pInstance);
    if (r != VK_SUCCESS && bumped) {
        Logf("vkCreateInstance with raised apiVersion failed (%d), retrying with the original info", (int)r);
        chain->u.pLayerInfo = savedLink;
        r = createInstance(pCreateInfo, pAllocator, pInstance);
    }
    if (r != VK_SUCCESS) { Logf("vkCreateInstance failed: %d", (int)r); return r; }

    auto* d = new InstanceData();
    d->instance = *pInstance;
    d->gipa = gipa;
#define LOADI(fn) d->fn = (PFN_vk##fn)gipa(*pInstance, "vk" #fn)
    LOADI(DestroyInstance);
    LOADI(CreateDevice);
    LOADI(GetPhysicalDeviceProperties);
    LOADI(EnumerateDeviceExtensionProperties);
    LOADI(GetPhysicalDeviceFeatures2);
    LOADI(GetPhysicalDeviceProperties2);
    LOADI(GetPhysicalDeviceMemoryProperties);
#undef LOADI
    if (!d->GetPhysicalDeviceFeatures2)
        d->GetPhysicalDeviceFeatures2 = (PFN_vkGetPhysicalDeviceFeatures2)gipa(*pInstance, "vkGetPhysicalDeviceFeatures2KHR");
    {
        std::lock_guard<std::mutex> lk(g_mapMtx);
        g_instances[Key(*pInstance)] = d;
    }
    Logf("vkCreateInstance OK");
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL SkyRT_DestroyInstance(VkInstance instance, const VkAllocationCallbacks* pAllocator) {
    g_stage = "vkDestroyInstance";
    InstanceData* d = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_mapMtx);
        auto it = g_instances.find(Key(instance));
        if (it != g_instances.end()) { d = it->second; g_instances.erase(it); }
    }
    Logf("vkDestroyInstance");
    if (!d) return;
    PFN_vkDestroyInstance fn = d->DestroyInstance;
    delete d;
    fn(instance, pAllocator);
}

// ============================================================ device
static VKAPI_ATTR VkResult VKAPI_CALL SkyRT_CreateDevice(VkPhysicalDevice phys, const VkDeviceCreateInfo* pCreateInfo,
                                                         const VkAllocationCallbacks* pAllocator, VkDevice* pDevice) {
    g_stage = "vkCreateDevice";
    InstanceData* inst = GetInst(Key(phys));
    if (!inst) return VK_ERROR_INITIALIZATION_FAILED;

    auto* chain = (VkLayerDeviceCreateInfo*)pCreateInfo->pNext;
    while (chain && !(chain->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO &&
                      chain->function == VK_LAYER_LINK_INFO))
        chain = (VkLayerDeviceCreateInfo*)chain->pNext;
    if (!chain) { Logf("ERROR: no loader link info in vkCreateDevice chain"); return VK_ERROR_INITIALIZATION_FAILED; }

    PFN_vkGetInstanceProcAddr gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr gdpa = chain->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;
    auto* savedLink = chain->u.pLayerInfo;

    // the loader's callback that gives our own command buffers a valid dispatch pointer
    PFN_vkSetDeviceLoaderData setLoaderData = nullptr;
    for (auto* c = (VkLayerDeviceCreateInfo*)pCreateInfo->pNext; c; c = (VkLayerDeviceCreateInfo*)c->pNext) {
        if (c->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO && c->function == VK_LOADER_DATA_CALLBACK) {
            setLoaderData = c->u.pfnSetDeviceLoaderData;
            break;
        }
    }

    auto createDevice = (PFN_vkCreateDevice)gipa(inst->instance, "vkCreateDevice");
    if (!createDevice) return VK_ERROR_INITIALIZATION_FAILED;

    VkPhysicalDeviceProperties props{};
    inst->GetPhysicalDeviceProperties(phys, &props);
    char apiS[32];
    ApiStr(props.apiVersion, apiS, sizeof apiS);
    Logf("GPU: %s | api %s | vendor 0x%x device 0x%x | driver 0x%x", props.deviceName, apiS, props.vendorID,
         props.deviceID, props.driverVersion);
    for (uint32_t i = 0; i < pCreateInfo->enabledExtensionCount; ++i)
        Logf("  device ext: %s", pCreateInfo->ppEnabledExtensionNames[i]);
    for (auto* p = (const VkBaseInStructure*)pCreateInfo->pNext; p; p = p->pNext)
        Logf("  device pNext sType=%d", (int)p->sType);

    // ---- decide whether RT can be enabled
    bool wantRT = !g_logOnly;
    const char* needExt[] = {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, VK_KHR_RAY_QUERY_EXTENSION_NAME,
                             VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};
    if (wantRT && props.apiVersion < kApi12) { Logf("RT skipped: device api < 1.2"); wantRT = false; }
    if (wantRT) {
        uint32_t n = 0;
        inst->EnumerateDeviceExtensionProperties(phys, nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> avail(n);
        inst->EnumerateDeviceExtensionProperties(phys, nullptr, &n, avail.data());
        for (const char* e : needExt) {
            bool found = false;
            for (auto& a : avail) if (!strcmp(a.extensionName, e)) { found = true; break; }
            if (!found) { Logf("RT skipped: device extension %s not available", e); wantRT = false; }
        }
    }
    if (wantRT) {
        if (!inst->GetPhysicalDeviceFeatures2) { Logf("RT skipped: no vkGetPhysicalDeviceFeatures2"); wantRT = false; }
        else {
            VkPhysicalDeviceBufferDeviceAddressFeatures bdaF{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
            VkPhysicalDeviceAccelerationStructureFeaturesKHR asF{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
            VkPhysicalDeviceRayQueryFeaturesKHR rqF{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
            VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            f2.pNext = &bdaF; bdaF.pNext = &asF; asF.pNext = &rqF;
            inst->GetPhysicalDeviceFeatures2(phys, &f2);
            Logf("GPU RT features: accelerationStructure=%u rayQuery=%u bufferDeviceAddress=%u", asF.accelerationStructure,
                 rqF.rayQuery, bdaF.bufferDeviceAddress);
            if (!asF.accelerationStructure || !rqF.rayQuery || !bdaF.bufferDeviceAddress) wantRT = false;
        }
    }

    // ---- build the modified create info
    VkDeviceCreateInfo ci = *pCreateInfo;
    std::vector<const char*> exts(pCreateInfo->ppEnabledExtensionNames,
                                  pCreateInfo->ppEnabledExtensionNames + pCreateInfo->enabledExtensionCount);
    VkPhysicalDeviceBufferDeviceAddressFeatures bdaAdd{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR asAdd{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceRayQueryFeaturesKHR rqAdd{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    std::vector<std::pair<VkBool32*, VkBool32>> undo;  // to restore the game's structs on fallback

    if (wantRT) {
        bool hasBda = false, hasV12 = false, hasAs = false, hasRq = false;
        auto setTrue = [&](VkBool32* f) { undo.emplace_back(f, *f); *f = VK_TRUE; };
        for (auto* p = (VkBaseOutStructure*)pCreateInfo->pNext; p; p = p->pNext) {
            switch (p->sType) {
                case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES:
                    setTrue(&((VkPhysicalDeviceVulkan12Features*)p)->bufferDeviceAddress); hasV12 = true; break;
                case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES:
                    setTrue(&((VkPhysicalDeviceBufferDeviceAddressFeatures*)p)->bufferDeviceAddress); hasBda = true; break;
                case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR:
                    setTrue(&((VkPhysicalDeviceAccelerationStructureFeaturesKHR*)p)->accelerationStructure); hasAs = true; break;
                case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR:
                    setTrue(&((VkPhysicalDeviceRayQueryFeaturesKHR*)p)->rayQuery); hasRq = true; break;
                default: break;
            }
        }
        if (!hasV12 && !hasBda) { bdaAdd.bufferDeviceAddress = VK_TRUE; bdaAdd.pNext = (void*)ci.pNext; ci.pNext = &bdaAdd; }
        if (!hasAs) { asAdd.accelerationStructure = VK_TRUE; asAdd.pNext = (void*)ci.pNext; ci.pNext = &asAdd; }
        if (!hasRq) { rqAdd.rayQuery = VK_TRUE; rqAdd.pNext = (void*)ci.pNext; ci.pNext = &rqAdd; }
        for (const char* e : needExt) {
            bool have = false;
            for (const char* x : exts) if (!strcmp(x, e)) { have = true; break; }
            if (!have) exts.push_back(e);
        }
        ci.enabledExtensionCount = (uint32_t)exts.size();
        ci.ppEnabledExtensionNames = exts.data();
        Logf("trying vkCreateDevice with RT extensions + features");
    }

    // ---- step 13: make sure samplerAnisotropy is enabled (the game may not ask for it)
    bool anisoWanted = false;
    VkPhysicalDeviceFeatures anisoFeat{};
    if (g_aniso > 1 && inst->GetPhysicalDeviceFeatures2) {
        VkPhysicalDeviceFeatures2 hw{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        inst->GetPhysicalDeviceFeatures2(phys, &hw);
        if (hw.features.samplerAnisotropy) {
            anisoWanted = true;
            bool done = false;
            for (auto* p = (VkBaseOutStructure*)pCreateInfo->pNext; p; p = p->pNext) {
                if (p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) {
                    VkBool32* f = &((VkPhysicalDeviceFeatures2*)p)->features.samplerAnisotropy;
                    undo.emplace_back(f, *f); *f = VK_TRUE; done = true; break;
                }
            }
            if (!done) {
                if (pCreateInfo->pEnabledFeatures) anisoFeat = *pCreateInfo->pEnabledFeatures;
                anisoFeat.samplerAnisotropy = VK_TRUE;
                ci.pEnabledFeatures = &anisoFeat;
            }
        }
    }
    VkResult r = createDevice(phys, &ci, pAllocator, pDevice);
    if (r != VK_SUCCESS && anisoWanted) {
        Logf("!!! vkCreateDevice with samplerAnisotropy failed (%d) - retrying without it", (int)r);
        anisoWanted = false;
        ci.pEnabledFeatures = pCreateInfo->pEnabledFeatures;
        for (auto& u : undo) if (u.first) { /* keep RT toggles; only the anisotropy flag matters here */ }
        r = createDevice(phys, &ci, pAllocator, pDevice);
    }
    if (r != VK_SUCCESS && wantRT) {
        Logf("!!! vkCreateDevice with RT failed (%d) - retrying with the game's original info (RT off)", (int)r);
        for (auto& u : undo) *u.first = u.second;
        chain->u.pLayerInfo = savedLink;
        wantRT = false;
        anisoWanted = false;  // the original info does not enable samplerAnisotropy
        r = createDevice(phys, pCreateInfo, pAllocator, pDevice);
    }
    if (r != VK_SUCCESS) { Logf("vkCreateDevice failed: %d", (int)r); return r; }

    auto* d = new DeviceData();
    d->device = *pDevice;
    d->inst = inst;
    d->phys = phys;
    d->gdpa = gdpa;
    d->rtEnabled = wantRT;
    d->setLoaderData = setLoaderData;
    d->anisoOk = anisoWanted;
    d->maxAniso = props.limits.maxSamplerAnisotropy;
#define LOADD(fn) d->fn = (PFN_vk##fn)gdpa(*pDevice, "vk" #fn)
    LOADD(DestroyDevice);
    LOADD(CreateBuffer);
    LOADD(AllocateMemory);
    LOADD(QueuePresentKHR);
    LOADD(QueueSubmit);
    LOADD(CmdDrawIndexed);
    LOADD(CmdDraw);
    LOADD(CmdDrawIndexedIndirect);
    LOADD(CmdDrawIndirect);
    LOADD(CreateSampler);
    LOADD(CmdDrawIndirectCount);
    if (!d->CmdDrawIndirectCount) d->CmdDrawIndirectCount = (PFN_vkCmdDrawIndirectCount)gdpa(*pDevice, "vkCmdDrawIndirectCountKHR");
    LOADD(CmdDrawIndexedIndirectCount);
    if (!d->CmdDrawIndexedIndirectCount) d->CmdDrawIndexedIndirectCount = (PFN_vkCmdDrawIndexedIndirectCount)gdpa(*pDevice, "vkCmdDrawIndexedIndirectCountKHR");
    LOADD(CmdExecuteCommands);
    LOADD(BindBufferMemory);
    LOADD(BindBufferMemory2);
    LOADD(MapMemory);
    LOADD(UnmapMemory);
    LOADD(FreeMemory);
    LOADD(DestroyBuffer);
    LOADD(GetBufferDeviceAddress);
    LOADD(BeginCommandBuffer);
    LOADD(CmdBindVertexBuffers);
    LOADD(CmdBindIndexBuffer);
    LOADD(CreateGraphicsPipelines);
    LOADD(DestroyPipeline);
    LOADD(CmdBindPipeline);
    LOADD(CreateImage);
    LOADD(CreateRenderPass);
    LOADD(CreateRenderPass2);
    LOADD(CmdBeginRenderPass);
    LOADD(CmdBeginRenderPass2);
    LOADD(GetDeviceQueue);
    LOADD(GetDeviceQueue2);
    LOADD(GetBufferMemoryRequirements);
    LOADD(CreateCommandPool);
    LOADD(DestroyCommandPool);
    LOADD(AllocateCommandBuffers);
    LOADD(EndCommandBuffer);
    LOADD(CreateFence);
    LOADD(DestroyFence);
    LOADD(WaitForFences);
    LOADD(GetAccelerationStructureBuildSizesKHR);
    LOADD(CreateAccelerationStructureKHR);
    LOADD(DestroyAccelerationStructureKHR);
    LOADD(CmdBuildAccelerationStructuresKHR);
    LOADD(CreateImageView);
    LOADD(CreateFramebuffer);
    LOADD(CmdEndRenderPass);
    LOADD(CreateDescriptorSetLayout);
    LOADD(CreatePipelineLayout);
    LOADD(CreateShaderModule);
    LOADD(DestroyShaderModule);
    LOADD(CreateComputePipelines);
    LOADD(CreateDescriptorPool);
    LOADD(AllocateDescriptorSets);
    LOADD(UpdateDescriptorSets);
    LOADD(CmdBindDescriptorSets);
    LOADD(CmdPushConstants);
    LOADD(CmdDispatch);
    LOADD(CmdPipelineBarrier);
    LOADD(CmdFillBuffer);
    LOADD(CmdCopyBuffer);
    LOADD(DestroyImageView);
    LOADD(DestroyImage);
    LOADD(GetAccelerationStructureDeviceAddressKHR);
    LOADD(GetImageMemoryRequirements);
    LOADD(DeviceWaitIdle);
    LOADD(BindImageMemory);
#undef LOADD
    d->track = wantRT && !g_logOnly && !g_noPaint;
    if (!d->GetBufferDeviceAddress)
        d->GetBufferDeviceAddress = (PFN_vkGetBufferDeviceAddress)gdpa(*pDevice, "vkGetBufferDeviceAddressKHR");
    {
        std::lock_guard<std::mutex> lk(g_mapMtx);
        g_devices[Key(*pDevice)] = d;
    }
    Logf("vkCreateDevice OK | RT %s", wantRT ? "ENABLED (ray_query + acceleration_structure + BDA)" : "off");
    return VK_SUCCESS;
}

static std::atomic<uint64_t> g_cDrawInd{0}, g_cDrawIndCnt{0}, g_cDrawIdxIndCnt{0}, g_cExec{0}, g_cDraw{0};
static void PollKeys(uint64_t frame);
static void PrintStats(DeviceData* d, const char* tag) {
    uint64_t frames = d->frames.load();
    uint64_t span = frames - d->lastStatFrame;
    if (!span) span = 1;
    d->lastStatFrame = frames;
    double f = (double)span;
    uint64_t di = d->drawIndexed.exchange(0), ic = d->indirectCalls.exchange(0);
    uint64_t isd = d->indirectSubdraws.exchange(0), sub = d->submits.exchange(0);
    Logf("[%s] frame %llu | per frame: drawIndexed %.0f, indirectCalls %.1f (subdraws %.0f), submits %.1f | "
         "buffers vtx=%llu idx=%llu ind=%llu patched=%llu fail=%llu | memory patched=%llu fail=%llu | RT=%s",
         tag, (unsigned long long)frames, di / f, ic / f, isd / f, sub / f, (unsigned long long)d->vtxBuf.load(),
         (unsigned long long)d->idxBuf.load(), (unsigned long long)d->indBuf.load(),
         (unsigned long long)d->bufPatched.load(), (unsigned long long)d->bufPatchFail.load(),
         (unsigned long long)d->memPatched.load(), (unsigned long long)d->memPatchFail.load(),
         d->rtEnabled ? "on" : "off");
    Logf("[%s] other draw APIs per frame: vkCmdDraw %.1f, DrawIndirect %.1f, DrawIndirectCount %.1f, DrawIndexedIndirectCount %.1f, ExecuteCommands %.1f",
         tag, g_cDraw.exchange(0) / f, g_cDrawInd.exchange(0) / f, g_cDrawIndCnt.exchange(0) / f, g_cDrawIdxIndCnt.exchange(0) / f, g_cExec.exchange(0) / f);
    {
        const uint64_t df = d->dynFrames.exchange(0), dg = d->dynGeomsSum.exchange(0), dt = d->dynTrisSum.exchange(0), dsk = d->dynSkipped.exchange(0);
        if (df || dsk) Logf("[%s] dynamic geometry (CPU-written vertex buffers): %llu frames with it, avg %.0f geoms / %.0f tris per frame, skipped %llu", tag, (unsigned long long)df, df ? (double)dg / df : 0.0, df ? (double)dt / df : 0.0, (unsigned long long)dsk);
    }
    Logf("[%s] rt passes injected %llu times | AS built %llu, frames skipped %llu, no-camera %llu | cache: %llu geometries, %llu tris, BLAS %llu KB | rebuilds %llu, evicted %llu",
         tag, (unsigned long long)d->paintInjected.load(), (unsigned long long)d->rtBuilt.load(),
         (unsigned long long)d->rtSkipped.load(), (unsigned long long)d->rtNoCam.load(), (unsigned long long)d->rtLastGeoms.load(),
         (unsigned long long)d->rtLastTris.load(), (unsigned long long)d->rtLastAsKB.load(),
         (unsigned long long)d->rtRebuilds.load(), (unsigned long long)d->rtEvicted.load());
    if (g_heapCount) {
        char line[512] = {};
        int off = 0;
        for (uint32_t i = 0; i < g_heapCount && off < 400; ++i)
            off += snprintf(line + off, sizeof line - off, " heap%u: live %.0f / peak %.0f / size %.0f MB |", i, g_heapLive[i].load() / 1048576.0,
                            g_heapPeak[i].load() / 1048576.0, g_heapSize[i] / 1048576.0);
        Logf("[%s] game memory:%s allocations >=256 MB: %llu, failed allocations: %llu", tag, line, (unsigned long long)g_allocBig.load(),
             (unsigned long long)g_allocFails.load());
    }
    {
        const uint64_t n = d->injectN.exchange(0);
        const uint64_t ui = d->usInject.exchange(0), um = d->usInjectMax.exchange(0), ud = d->usDraw.exchange(0);
        if (n) Logf("[%s] layer CPU cost per frame: RT pass recording avg %.0f us (max %.0f us), geometry collection avg %.0f us", tag,
                    (double)ui / (double)n, (double)um, (double)ud / (double)n);
    }
}

static VKAPI_ATTR void VKAPI_CALL SkyRT_DestroyDevice(VkDevice device, const VkAllocationCallbacks* pAllocator) {
    g_stage = "vkDestroyDevice";
    DeviceData* d = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_mapMtx);
        auto it = g_devices.find(Key(device));
        if (it != g_devices.end()) { d = it->second; g_devices.erase(it); }
    }
    if (!d) return;
    PrintStats(d, "final");
    Logf("vkDestroyDevice");
    PFN_vkDestroyDevice fn = d->DestroyDevice;
    delete d;
    fn(device, pAllocator);
}

// ============================================================ buffers / memory
static void RegisterBuf(DeviceData* d, VkBuffer b, VkDeviceSize size, VkBufferUsageFlags usage) {
    std::lock_guard<std::mutex> lk(d->regMtx);
    d->bufs[b] = DeviceData::BufInfo{size, usage, VK_NULL_HANDLE, 0};
}

static VkPhysicalDeviceMemoryProperties g_memProps{};
static uint32_t HeapOfType(DeviceData* d, uint32_t type) {
    static std::once_flag once;
    VkPhysicalDeviceMemoryProperties& mp = g_memProps;
    std::call_once(once, [&] {
        d->inst->GetPhysicalDeviceMemoryProperties(d->phys, &mp);
        g_heapCount = std::min<uint32_t>(mp.memoryHeapCount, 16);
        for (uint32_t i = 0; i < g_heapCount; ++i) g_heapSize[i] = mp.memoryHeaps[i].size;
    });
    return type < mp.memoryTypeCount ? mp.memoryTypes[type].heapIndex : 0xFFFFFFFFu;
}

static void RegisterMem(DeviceData* d, VkDeviceMemory m, VkDeviceSize size, uint32_t type = 0xFFFFFFFFu) {
    uint32_t heap = type == 0xFFFFFFFFu ? 0xFFFFFFFFu : HeapOfType(d, type);
    if (heap < 16) {
        const uint64_t now = g_heapLive[heap] += size;
        uint64_t pk = g_heapPeak[heap].load();
        while (now > pk && !g_heapPeak[heap].compare_exchange_weak(pk, now)) {}
        if (size >= (256ull << 20)) g_allocBig++;
    }
    std::lock_guard<std::mutex> lk(d->regMtx);
    DeviceData::MemInfo mi{size, nullptr, 0, 0};
    mi.heap = heap;
    mi.hostVisible = type < g_memProps.memoryTypeCount && (g_memProps.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    d->mems[m] = mi;
}

static VKAPI_ATTR VkResult VKAPI_CALL SkyRT_CreateBuffer(VkDevice device, const VkBufferCreateInfo* pCreateInfo,
                                                         const VkAllocationCallbacks* pAllocator, VkBuffer* pBuffer) {
    DeviceData* d = GetDev(Key(device));
    if (!d) return VK_ERROR_INITIALIZATION_FAILED;
    g_stage = "vkCreateBuffer";

    const VkBufferUsageFlags u = pCreateInfo->usage;
    const bool isVtx = (u & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) != 0;
    const bool isIdx = (u & VK_BUFFER_USAGE_INDEX_BUFFER_BIT) != 0;
    const bool isInd = (u & VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT) != 0;
    const bool isUbo = (u & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) != 0;
    if (isVtx) d->vtxBuf++;
    if (isIdx) d->idxBuf++;
    if (isInd) d->indBuf++;

    static std::atomic<int> logged{0};
    auto logBuf = [&](VkBuffer b, const char* note) {
        if (!(isVtx || isIdx || isInd)) return;
        if (logged.fetch_add(1) >= kMaxBufferLogLines) return;
        Logf("buffer %p size=%llu usage=0x%x [%s%s%s] %s", (void*)b, (unsigned long long)pCreateInfo->size, (unsigned)u,
             isVtx ? "VTX " : "", isIdx ? "IDX " : "", isInd ? "IND" : "", note);
    };

    if (d->rtEnabled && (isVtx || isIdx || isInd)) {
        VkBufferCreateInfo ci = *pCreateInfo;
        ci.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (isVtx || isIdx) ci.usage |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        VkResult r = d->CreateBuffer(device, &ci, pAllocator, pBuffer);
        if (r == VK_SUCCESS) {
            d->bufPatched++;
            RegisterBuf(d, *pBuffer, pCreateInfo->size, u);
            logBuf(*pBuffer, "patched (+device address, +storage, +AS input)");
            return r;
        }
        d->bufPatchFail++;
        Logf("!!! patched vkCreateBuffer failed (%d), falling back to the original", (int)r);
    }
    VkResult r = d->CreateBuffer(device, pCreateInfo, pAllocator, pBuffer);
    if (r == VK_SUCCESS) {
        if (isVtx || isIdx || isInd || isUbo) RegisterBuf(d, *pBuffer, pCreateInfo->size, u);
        logBuf(*pBuffer, "not patched");
    }
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL SkyRT_AllocateMemory(VkDevice device, const VkMemoryAllocateInfo* pAllocateInfo,
                                                           const VkAllocationCallbacks* pAllocator,
                                                           VkDeviceMemory* pMemory) {
    DeviceData* d = GetDev(Key(device));
    if (!d) return VK_ERROR_INITIALIZATION_FAILED;
    g_stage = "vkAllocateMemory";

    if (d->rtEnabled) {
        VkMemoryAllocateInfo ci = *pAllocateInfo;
        VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        VkMemoryAllocateFlagsInfo* existing = nullptr;
        for (auto* p = (VkBaseOutStructure*)pAllocateInfo->pNext; p; p = p->pNext)
            if (p->sType == VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO) existing = (VkMemoryAllocateFlagsInfo*)p;
        VkMemoryAllocateFlags oldFlags = 0;
        if (existing) {
            oldFlags = existing->flags;
            existing->flags |= VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        } else {
            fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
            fi.pNext = pAllocateInfo->pNext;
            ci.pNext = &fi;
        }
        VkResult r = d->AllocateMemory(device, &ci, pAllocator, pMemory);
        if (r == VK_SUCCESS) {
            d->memPatched++;
            RegisterMem(d, *pMemory, pAllocateInfo->allocationSize, pAllocateInfo->memoryTypeIndex);
            return r;
        }
        if (existing) existing->flags = oldFlags;
        if (r == VK_ERROR_OUT_OF_DEVICE_MEMORY || r == VK_ERROR_OUT_OF_HOST_MEMORY) {  // real OOM
            g_allocFails++;
            const uint32_t hp = HeapOfType(d, pAllocateInfo->memoryTypeIndex);
            Logf("!!! vkAllocateMemory OUT OF MEMORY (%d): asked %.1f MB, type %u, heap %u (live %.0f MB of %.0f MB)", (int)r,
                 pAllocateInfo->allocationSize / 1048576.0, pAllocateInfo->memoryTypeIndex, hp,
                 hp < 16 ? g_heapLive[hp].load() / 1048576.0 : 0.0, hp < 16 ? g_heapSize[hp] / 1048576.0 : 0.0);
            FlushToDisk();
            return r;
        }
        d->memPatchFail++;
        static std::atomic<int> failLogs{0};
        if (failLogs.fetch_add(1) < 20)
            Logf("!!! patched vkAllocateMemory failed (%d) size=%llu type=%u, falling back", (int)r,
                 (unsigned long long)pAllocateInfo->allocationSize, pAllocateInfo->memoryTypeIndex);
    }
    VkResult r2 = d->AllocateMemory(device, pAllocateInfo, pAllocator, pMemory);
    if (r2 == VK_SUCCESS) RegisterMem(d, *pMemory, pAllocateInfo->allocationSize, pAllocateInfo->memoryTypeIndex);
    else {
        g_allocFails++;
        const uint32_t hp = HeapOfType(d, pAllocateInfo->memoryTypeIndex);
        Logf("!!! vkAllocateMemory FAILED (%d): asked %.1f MB, type %u, heap %u (live %.0f MB of %.0f MB)", (int)r2,
             pAllocateInfo->allocationSize / 1048576.0, pAllocateInfo->memoryTypeIndex, hp,
             hp < 16 ? g_heapLive[hp].load() / 1048576.0 : 0.0, hp < 16 ? g_heapSize[hp] / 1048576.0 : 0.0);
        FlushToDisk();
    }
    return r2;
}

// ============================================================ bind / map tracking
static VKAPI_ATTR VkResult VKAPI_CALL SkyRT_BindBufferMemory(VkDevice device, VkBuffer buffer, VkDeviceMemory memory,
                                                             VkDeviceSize memoryOffset) {
    DeviceData* d = GetDev(Key(device));
    if (!d) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = d->BindBufferMemory(device, buffer, memory, memoryOffset);
    if (r == VK_SUCCESS) {
        d->bindCalls1++;
        std::lock_guard<std::mutex> lk(d->regMtx);
        auto it = d->bufs.find(buffer);
        if (it != d->bufs.end()) { it->second.mem = memory; it->second.memOff = memoryOffset; }
    }
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL SkyRT_BindBufferMemory2(VkDevice device, uint32_t bindInfoCount,
                                                              const VkBindBufferMemoryInfo* pBindInfos) {
    DeviceData* d = GetDev(Key(device));
    if (!d) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = d->BindBufferMemory2(device, bindInfoCount, pBindInfos);
    if (r == VK_SUCCESS) {
        d->bindCalls2++;
        std::lock_guard<std::mutex> lk(d->regMtx);
        for (uint32_t i = 0; i < bindInfoCount; ++i) {
            auto it = d->bufs.find(pBindInfos[i].buffer);
            if (it != d->bufs.end()) { it->second.mem = pBindInfos[i].memory; it->second.memOff = pBindInfos[i].memoryOffset; }
        }
    }
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL SkyRT_MapMemory(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset,
                                                      VkDeviceSize size, VkMemoryMapFlags flags, void** ppData) {
    DeviceData* d = GetDev(Key(device));
    if (!d) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = d->MapMemory(device, memory, offset, size, flags, ppData);
    if (r == VK_SUCCESS && ppData && *ppData) {
        std::lock_guard<std::mutex> lk(d->regMtx);
        auto it = d->mems.find(memory);
        if (it != d->mems.end()) {
            it->second.mapped = (char*)*ppData;
            it->second.mapOffset = offset;
            it->second.mapSize = (size == VK_WHOLE_SIZE) ? (it->second.size - offset) : size;
        }
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL SkyRT_UnmapMemory(VkDevice device, VkDeviceMemory memory) {
    DeviceData* d = GetDev(Key(device));
    if (!d) return;
    {
        std::lock_guard<std::mutex> lk(d->regMtx);
        auto it = d->mems.find(memory);
        if (it != d->mems.end()) { it->second.mapped = nullptr; it->second.mapSize = 0; }
    }
    d->UnmapMemory(device, memory);
}

static VKAPI_ATTR void VKAPI_CALL SkyRT_FreeMemory(VkDevice device, VkDeviceMemory memory,
                                                   const VkAllocationCallbacks* pAllocator) {
    DeviceData* d = GetDev(Key(device));
    if (!d) return;
    {
        std::lock_guard<std::mutex> lk(d->regMtx);
        auto mit = d->mems.find(memory);
        if (mit != d->mems.end()) {
            if (mit->second.heap < 16) g_heapLive[mit->second.heap] -= std::min<uint64_t>(g_heapLive[mit->second.heap].load(), mit->second.size);
            d->mems.erase(mit);
        }
    }
    {   // an AS build recorded in a command buffer that has not run yet may still read this memory:
        // keep the memory alive for a few frames instead of freeing it now
        std::lock_guard<std::mutex> lk(d->cacheMtx);
        if (d->buildMems.erase(memory)) {
            for (auto it = d->geoCache.begin(); it != d->geoCache.end();) {
                if (it->second.vm == memory || it->second.im == memory) {
                    d->cacheTris -= std::min<uint64_t>(d->cacheTris, it->second.r.primitiveCount);
                    it = d->geoCache.erase(it);
                    d->rtEvicted++;
                } else ++it;
            }
            d->cacheDirty = true;
            d->deferred.push_back({VK_NULL_HANDLE, memory, d->frames.load()});
            d->rtDeferred++;
            return;
        }
    }
    d->FreeMemory(device, memory, pAllocator);
}

static VKAPI_ATTR void VKAPI_CALL SkyRT_DestroyBuffer(VkDevice device, VkBuffer buffer,
                                                      const VkAllocationCallbacks* pAllocator) {
    DeviceData* d = GetDev(Key(device));
    if (!d) return;
    {
        std::lock_guard<std::mutex> lk(d->regMtx);
        d->bufs.erase(buffer);
    }
    {
        std::lock_guard<std::mutex> lk(d->cacheMtx);
        if (d->cachedBufs.erase(buffer)) {
            for (auto it = d->geoCache.begin(); it != d->geoCache.end();) {
                if (it->first.vb == buffer || it->first.ib == buffer) {
                    d->cacheTris -= std::min<uint64_t>(d->cacheTris, it->second.r.primitiveCount);
                    it = d->geoCache.erase(it);
                    d->rtEvicted++;
                } else ++it;
            }
            d->cacheDirty = true;
            d->deferred.push_back({buffer, VK_NULL_HANDLE, d->frames.load()});
            d->rtDeferred++;
            return;
        }
    }
    d->DestroyBuffer(device, buffer, pAllocator);
}

// ============================================================ probe (runs from vkQueuePresentKHR)
static bool SafeCopy(void* dst, const void* src, size_t n) {
    __try { memcpy(dst, src, n); return true; }
    __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
}

static void DumpIndirect(DeviceData* d, const DeviceData::BufInfo& bi) {
    DeviceData::MemInfo mi{};
    bool found = false;
    {
        std::lock_guard<std::mutex> lk(d->regMtx);
        auto it = d->mems.find(bi.mem);
        if (it != d->mems.end()) { mi = it->second; found = true; }
    }
    if (!found || !mi.mapped) {
        Logf("  indirect buffer is NOT host-mapped right now (GPU-written, or the game maps it on demand)");
        return;
    }
    if (bi.memOff < mi.mapOffset || (bi.memOff - mi.mapOffset) + bi.size > mi.mapSize) {
        Logf("  indirect buffer lies outside the mapped range (memOff=%llu mapOffset=%llu mapSize=%llu)",
             (unsigned long long)bi.memOff, (unsigned long long)mi.mapOffset, (unsigned long long)mi.mapSize);
        return;
    }
    const size_t n = (size_t)(bi.size / sizeof(VkDrawIndexedIndirectCommand));
    std::vector<VkDrawIndexedIndirectCommand> cmds(n);
    if (!SafeCopy(cmds.data(), mi.mapped + (bi.memOff - mi.mapOffset), n * sizeof(VkDrawIndexedIndirectCommand))) {
        Logf("  reading the mapped indirect buffer faulted");
        return;
    }
    size_t valid = 0, shown = 0, multi = 0;
    uint32_t maxInst = 0, maxFirstInst = 0;
    int32_t maxVo = 0;
    for (size_t i = 0; i < n; ++i) {
        const VkDrawIndexedIndirectCommand& c = cmds[i];
        if (!c.indexCount || !c.instanceCount) continue;
        valid++;
        if (c.instanceCount > 1) multi++;
        if (c.instanceCount > maxInst) maxInst = c.instanceCount;
        if (c.firstInstance > maxFirstInst) maxFirstInst = c.firstInstance;
        if (c.vertexOffset > maxVo) maxVo = c.vertexOffset;
        if (shown < 12) {
            Logf("  cmd[%zu] indexCount=%u instanceCount=%u firstIndex=%u vertexOffset=%d firstInstance=%u", i, c.indexCount,
                 c.instanceCount, c.firstIndex, (int)c.vertexOffset, c.firstInstance);
            shown++;
        }
    }
    Logf("  indirect buffer: %zu slots, %zu look valid, %zu with instanceCount>1 (max %u), max firstInstance %u, max vertexOffset %d",
         n, valid, multi, maxInst, maxFirstInst, (int)maxVo);
}

static void ProbeBuffers(DeviceData* d) {
    g_stage = "ProbeBuffers";
    Logf("--- buffer probe --- vkBindBufferMemory calls: v1=%llu v2=%llu", (unsigned long long)d->bindCalls1.load(),
         (unsigned long long)d->bindCalls2.load());
    if (!d->rtEnabled || !d->GetBufferDeviceAddress) {
        Logf("probe skipped (RT off or no vkGetBufferDeviceAddress)");
        return;
    }
    struct Item { VkBuffer b; DeviceData::BufInfo i; };
    std::vector<Item> pick;
    uint64_t nBound = 0, nUnbound = 0;
    {
        std::lock_guard<std::mutex> lk(d->regMtx);
        int vs = 0, is = 0, vd = 0;
        for (auto& kv : d->bufs) {
            const DeviceData::BufInfo& i = kv.second;
            if (!i.mem) { nUnbound++; continue; }
            nBound++;
            const bool ind = (i.usage & VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT) != 0;
            const bool vtx = (i.usage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) != 0;
            const bool idx = (i.usage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT) != 0;
            const bool staged = (i.usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT) != 0;
            if (ind) pick.push_back(Item{kv.first, i});
            else if (vtx && staged && vs < 3) { pick.push_back(Item{kv.first, i}); vs++; }
            else if (idx && staged && is < 3) { pick.push_back(Item{kv.first, i}); is++; }
            else if (vtx && !staged && vd < 3) { pick.push_back(Item{kv.first, i}); vd++; }
        }
    }
    Logf("registry: %llu buffers bound, %llu not bound yet", (unsigned long long)nBound, (unsigned long long)nUnbound);
    for (auto& it : pick) {
        VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        ai.buffer = it.b;
        VkDeviceAddress addr = d->GetBufferDeviceAddress(d->device, &ai);
        const VkBufferUsageFlags u = it.i.usage;
        Logf("buf %p size=%llu usage=0x%x [%s%s%s%s] deviceAddress=0x%llx mem=%p memOff=%llu", (void*)it.b,
             (unsigned long long)it.i.size, (unsigned)u, (u & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) ? "VTX " : "",
             (u & VK_BUFFER_USAGE_INDEX_BUFFER_BIT) ? "IDX " : "", (u & VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT) ? "IND " : "",
             (u & VK_BUFFER_USAGE_TRANSFER_DST_BIT) ? "STAGED" : "", (unsigned long long)addr, (void*)it.i.mem,
             (unsigned long long)it.i.memOff);
        if (u & VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT) DumpIndirect(d, it.i);
    }
    Logf("--- end of probe ---");
}

// ============================================================ frame capture
static VKAPI_ATTR VkResult VKAPI_CALL SkyRT_BeginCommandBuffer(VkCommandBuffer cb,
                                                               const VkCommandBufferBeginInfo* pBeginInfo) {
    DeviceData* d = GetDev(Key(cb));
    if (!d) return VK_ERROR_INITIALIZATION_FAILED;
    if (d->capturing.load(std::memory_order_relaxed) || d->track) {
        std::lock_guard<std::mutex> lk(d->cmdMtx);
        d->cmdState.erase(cb);
        d->cmdDraws.erase(cb);
    }
    if (d->track) {
        {
            std::lock_guard<std::mutex> lk(d->geoMtx);
            d->frameGeo.erase(cb);
        }
        std::lock_guard<std::mutex> lk(d->uboMtx);
        d->mainUbo.erase(cb);
    }
    return d->BeginCommandBuffer(cb, pBeginInfo);
}

static VKAPI_ATTR void VKAPI_CALL SkyRT_CmdBindVertexBuffers(VkCommandBuffer cb, uint32_t firstBinding,
                                                             uint32_t bindingCount, const VkBuffer* pBuffers,
                                                             const VkDeviceSize* pOffsets) {
    DeviceData* d = GetDev(Key(cb));
    if (!d) return;
    if (d->track || d->capturing.load(std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> lk(d->cmdMtx);
        DeviceData::CmdState& s = d->cmdState[cb];
        for (uint32_t i = 0; i < bindingCount; ++i) {
            const uint32_t slot = firstBinding + i;
            if (slot < 8) { s.vb[slot] = pBuffers[i]; s.vbOff[slot] = pOffsets[i]; }
        }
    }
    d->CmdBindVertexBuffers(cb, firstBinding, bindingCount, pBuffers, pOffsets);
}

static VKAPI_ATTR void VKAPI_CALL SkyRT_CmdBindIndexBuffer(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset,
                                                           VkIndexType indexType) {
    DeviceData* d = GetDev(Key(cb));
    if (!d) return;
    if (d->track || d->capturing.load(std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> lk(d->cmdMtx);
        DeviceData::CmdState& s = d->cmdState[cb];
        s.ib = buffer;
        s.ibOff = offset;
        s.ibType = indexType;
    }
    d->CmdBindIndexBuffer(cb, buffer, offset, indexType);
}

static void RecordDraw(DeviceData* d, VkCommandBuffer cb, bool indirect, VkBuffer indBuf, VkDeviceSize indOff,
                       uint32_t drawCount, uint32_t stride, const VkDrawIndexedIndirectCommand& c,
                       uint64_t recHash = 0, bool recRead = false) {
    std::lock_guard<std::mutex> lk(d->cmdMtx);
    DeviceData::DrawRec r{};
    r.recHash = recHash;
    r.recRead = recRead;
    r.st = d->cmdState[cb];
    r.indirect = indirect;
    r.indBuf = indBuf;
    r.indOff = indOff;
    r.drawCount = drawCount;
    r.stride = stride;
    r.c = c;
    std::vector<DeviceData::DrawRec>& v = d->cmdDraws[cb];
    if (v.size() < 20000) v.push_back(r);
}

// reads indirect commands from host-mapped memory (the game fills the buffer on the CPU)
static bool ReadIndirect(DeviceData* d, VkBuffer buf, VkDeviceSize off, uint32_t count, uint32_t stride,
                         std::vector<VkDrawIndexedIndirectCommand>& out) {
    if (count > 65536) return false;
    DeviceData::BufInfo bi{};
    DeviceData::MemInfo mi{};
    {
        std::lock_guard<std::mutex> lk(d->regMtx);
        auto b = d->bufs.find(buf);
        if (b == d->bufs.end() || !b->second.mem) return false;
        bi = b->second;
        auto m = d->mems.find(bi.mem);
        if (m == d->mems.end() || !m->second.mapped) return false;
        mi = m->second;
    }
    if (bi.memOff < mi.mapOffset) return false;
    const VkDeviceSize base = bi.memOff - mi.mapOffset + off;
    const VkDeviceSize last = base + (VkDeviceSize)(count ? count - 1 : 0) * stride + sizeof(VkDrawIndexedIndirectCommand);
    if (last > mi.mapSize) return false;
    out.resize(count);
    for (uint32_t i = 0; i < count; ++i)
        if (!SafeCopy(&out[i], mi.mapped + base + (VkDeviceSize)i * stride, sizeof(VkDrawIndexedIndirectCommand)))
            return false;
    return true;
}

static uint64_t HashCmds(const std::vector<VkDrawIndexedIndirectCommand>& v) {
    uint64_t h = 1469598103934665603ull;
    for (const VkDrawIndexedIndirectCommand& c : v) {
        const uint32_t a[5] = {c.indexCount, c.instanceCount, c.firstIndex, (uint32_t)c.vertexOffset, c.firstInstance};
        for (uint32_t x : a) { h ^= x; h *= 1099511628211ull; }
    }
    return h;
}

// copies bytes out of a host-mapped buffer (works for buffers in the registry whose memory the game mapped)
static bool ReadBufHost(DeviceData* d, VkBuffer buf, VkDeviceSize off, void* out, size_t n) {
    DeviceData::BufInfo bi{};
    DeviceData::MemInfo mi{};
    {
        std::lock_guard<std::mutex> lk(d->regMtx);
        auto b = d->bufs.find(buf);
        if (b == d->bufs.end() || !b->second.mem) return false;
        bi = b->second;
        auto m = d->mems.find(bi.mem);
        if (m == d->mems.end() || !m->second.mapped) return false;
        mi = m->second;
    }
    if (off + n > bi.size || bi.memOff < mi.mapOffset) return false;
    const VkDeviceSize base = bi.memOff - mi.mapOffset + off;
    if (base + n > mi.mapSize) return false;
    return SafeCopy(out, mi.mapped + base, n);
}

// called from vkQueueSubmit while capturing: the CPU-side indirect data is final at this point
static void ResolveSubmit(DeviceData* d, uint32_t submitCount, const VkSubmitInfo* pSubmits) {
    for (uint32_t s = 0; s < submitCount; ++s) {
        for (uint32_t i = 0; i < pSubmits[s].commandBufferCount; ++i) {
            VkCommandBuffer cb = pSubmits[s].pCommandBuffers[i];
            std::vector<DeviceData::DrawRec> draws;
            {
                std::lock_guard<std::mutex> lk(d->cmdMtx);
                auto it = d->cmdDraws.find(cb);
                if (it == d->cmdDraws.end()) continue;
                draws.swap(it->second);
            }
            for (const DeviceData::DrawRec& r : draws) {
                if (!r.indirect) {
                    std::lock_guard<std::mutex> lk(d->cmdMtx);
                    if (d->resolved.size() < 200000) d->resolved.push_back(DeviceData::Resolved{r.st, r.c, false});
                    continue;
                }
                std::vector<VkDrawIndexedIndirectCommand> cmds;
                const uint32_t stride = r.stride ? r.stride : (uint32_t)sizeof(VkDrawIndexedIndirectCommand);
                const bool ok = ReadIndirect(d, r.indBuf, r.indOff, r.drawCount, stride, cmds);
                if (ok && r.recRead) {
                    d->recChecked++;
                    if (HashCmds(cmds) != r.recHash) d->recDiffer++;
                } else if (!r.recRead) {
                    d->recUnreadable++;
                }
                std::lock_guard<std::mutex> lk(d->cmdMtx);
                d->groups.push_back(DeviceData::Group{r.indBuf, r.indOff, r.drawCount, stride, r.st, ok});
                if (!ok) { d->unreadable++; continue; }
                for (const VkDrawIndexedIndirectCommand& c : cmds)
                    if (c.indexCount && c.instanceCount && d->resolved.size() < 200000)
                        d->resolved.push_back(DeviceData::Resolved{r.st, c, true});
            }
        }
    }
}

static unsigned long long BufSizeOf(DeviceData* d, VkBuffer b) {
    if (!b) return 0;
    std::lock_guard<std::mutex> lk(d->regMtx);
    auto it = d->bufs.find(b);
    return it == d->bufs.end() ? 0ull : (unsigned long long)it->second.size;
}

static void LogTop(DeviceData* d, const char* title, const std::unordered_map<VkBuffer, uint32_t>& use, size_t topN) {
    std::vector<std::pair<VkBuffer, uint32_t>> v(use.begin(), use.end());
    std::sort(v.begin(), v.end(), [](const std::pair<VkBuffer, uint32_t>& a, const std::pair<VkBuffer, uint32_t>& b) {
        return a.second > b.second;
    });
    Logf("  %s: %zu unique", title, v.size());
    for (size_t i = 0; i < v.size() && i < topN; ++i)
        Logf("    %p size=%llu used by %u draws", (void*)v[i].first, BufSizeOf(d, v[i].first), v[i].second);
}

// ------------------------------------------------------------ pipelines: static vertex layout
static VKAPI_ATTR VkResult VKAPI_CALL SkyRT_CreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo* pCreateInfo,
                                                                const VkAllocationCallbacks* pAllocator, VkShaderModule* pModule) {
    DeviceData* d = GetDev(Key(device));
    if (!d) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = d->CreateShaderModule(device, pCreateInfo, pAllocator, pModule);
    if (r == VK_SUCCESS && pCreateInfo && pCreateInfo->pCode) {
        uint64_t h = 1469598103934665603ull;
        const uint32_t n = (uint32_t)(pCreateInfo->codeSize / 4);
        for (uint32_t i = 0; i < n; ++i) { h ^= pCreateInfo->pCode[i]; h *= 1099511628211ull; }
        if (g_dumpShaders) {   // step 15: keep the shader so the skinning of characters can be analysed offline
            static std::mutex dm;
            static std::unordered_set<uint64_t> seen;
            std::lock_guard<std::mutex> dl(dm);
            if (seen.insert(h).second) {
                char dir[MAX_PATH], path[MAX_PATH + 64];
                ModuleDir(dir, sizeof dir);
                strncat(dir, "\\SkyRT_shaders", sizeof dir - strlen(dir) - 1);
                CreateDirectoryA(dir, nullptr);
                snprintf(path, sizeof path, "%s\\%016llx.spv", dir, (unsigned long long)h);
                FILE* f = fopen(path, "wb");
                if (f) { fwrite(pCreateInfo->pCode, 1, pCreateInfo->codeSize, f); fclose(f); }
                if (seen.size() == 1 || seen.size() % 50 == 0) Logf("rt: shader dump: %zu distinct shaders written to %s", seen.size(), dir);
            }
        }
        std::lock_guard<std::mutex> lk(d->regMtx);
        d->smHash[*pModule] = h;
    }
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL SkyRT_CreateGraphicsPipelines(VkDevice device, VkPipelineCache cache,
                                                                    uint32_t count,
                                                                    const VkGraphicsPipelineCreateInfo* pInfos,
                                                                    const VkAllocationCallbacks* pAllocator,
                                                                    VkPipeline* pPipelines) {
    DeviceData* d = GetDev(Key(device));
    if (!d) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = d->CreateGraphicsPipelines(device, cache, count, pInfos, pAllocator, pPipelines);
    if (r == VK_SUCCESS) {
        std::lock_guard<std::mutex> lk(d->regMtx);
        for (uint32_t i = 0; i < count; ++i) {
            if (!pPipelines[i]) continue;
            DeviceData::PipeInfo pi;
            const VkPipelineVertexInputStateCreateInfo* vi = pInfos[i].pVertexInputState;
            if (vi) {
                for (uint32_t b = 0; b < vi->vertexBindingDescriptionCount; ++b) {
                    const VkVertexInputBindingDescription& x = vi->pVertexBindingDescriptions[b];
                    pi.binds.push_back(DeviceData::PipeInfo::Bind{x.binding, x.stride, (uint32_t)x.inputRate});
                }
                for (uint32_t a = 0; a < vi->vertexAttributeDescriptionCount; ++a) {
                    const VkVertexInputAttributeDescription& x = vi->pVertexAttributeDescriptions[a];
                    pi.attrs.push_back(DeviceData::PipeInfo::Attr{x.location, x.binding, (uint32_t)x.format, x.offset});
                }
            }
            for (const DeviceData::PipeInfo::Bind& b : pi.binds) if (b.rate != 0) pi.instanced = true;
            {
                const VkPipelineDepthStencilStateCreateInfo* ds = pInfos[i].pDepthStencilState;
                const VkPipelineColorBlendStateCreateInfo* cbs = pInfos[i].pColorBlendState;
                bool blend = false;
                if (cbs) for (uint32_t k = 0; k < cbs->attachmentCount; ++k) if (cbs->pAttachments[k].blendEnable) blend = true;
                pi.opaque = ds && ds->depthWriteEnable && !blend;
                pi.blend = blend;
                pi.depthTest = ds && ds->depthTestEnable;
                pi.depthWrite = ds && ds->depthWriteEnable;
                if (cbs && cbs->attachmentCount) { pi.srcColor = (uint32_t)cbs->pAttachments[0].srcColorBlendFactor; pi.dstColor = (uint32_t)cbs->pAttachments[0].dstColorBlendFactor; }
                for (uint32_t k = 0; k < pInfos[i].stageCount; ++k) {
                    auto m = d->smHash.find(pInfos[i].pStages[k].module);
                    if (m == d->smHash.end()) continue;
                    if (pInfos[i].pStages[k].stage == VK_SHADER_STAGE_VERTEX_BIT) pi.vsHash = m->second;
                    if (pInfos[i].pStages[k].stage == VK_SHADER_STAGE_FRAGMENT_BIT) pi.fsHash = m->second;
                }
            }
            d->pipes[pPipelines[i]] = pi;
        }
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL SkyRT_DestroyPipeline(VkDevice device, VkPipeline pipeline,
                                                        const VkAllocationCallbacks* pAllocator) {
    DeviceData* d = GetDev(Key(device));
    if (!d) return;
    {
        std::lock_guard<std::mutex> lk(d->regMtx);
        d->pipes.erase(pipeline);
    }
    d->DestroyPipeline(device, pipeline, pAllocator);
}

static VKAPI_ATTR void VKAPI_CALL SkyRT_CmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bindPoint,
                                                        VkPipeline pipeline) {
    DeviceData* d = GetDev(Key(cb));
    if (!d) return;
    if (bindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS && (d->track || d->capturing.load(std::memory_order_relaxed))) {
        std::lock_guard<std::mutex> lk(d->cmdMtx);
        d->cmdState[cb].pipe = pipeline;
    }
    d->CmdBindPipeline(cb, bindPoint, pipeline);
}

static const char* FormatName(uint32_t f) {
    static char buf[16];
    switch (f) {
        case 16: return "R8G8_UNORM";
        case 44: return "B8G8R8A8_UNORM";
        case 50: return "B8G8R8A8_SRGB";
        case 64: return "A2B10G10R10_UNORM";
        case 91: return "R16G16B16A16_UNORM";
        case 122: return "B10G11R11_UFLOAT";
        case 124: return "D16_UNORM";
        case 126: return "D32_SFLOAT";
        case 129: return "D24_UNORM_S8_UINT";
        case 130: return "D32_SFLOAT_S8_UINT";
        case 37: return "R8G8B8A8_UNORM";
        case 38: return "R8G8B8A8_SNORM";
        case 41: return "R8G8B8A8_UINT";
        case 83: return "R16G16_SFLOAT";
        case 90: return "R16G16B16_SFLOAT";
        case 97: return "R16G16B16A16_SFLOAT";
        case 98: return "R32_UINT";
        case 100: return "R32_SFLOAT";
        case 103: return "R32G32_SFLOAT";
        case 104: return "R32G32B32_UINT";
        case 106: return "R32G32B32_SFLOAT";
        case 107: return "R32G32B32A32_UINT";
        case 109: return "R32G32B32A32_SFLOAT";
        default: snprintf(buf, sizeof buf, "fmt%u", f); return buf;
    }
}

// ------------------------------------------------------------ images / render passes (log only) + queues
static void LayoutStr(int l, char* out, size_t n) {
    switch (l) {
        case 0: snprintf(out, n, "UNDEFINED"); break;
        case 1: snprintf(out, n, "GENERAL"); break;
        case 2: snprintf(out, n, "COLOR_ATT"); break;
        case 3: snprintf(out, n, "DEPTH_STENCIL_ATT"); break;
        case 4: snprintf(out, n, "DEPTH_STENCIL_READ"); break;
        case 5: snprintf(out, n, "SHADER_READ"); break;
        case 6: snprintf(out, n, "TRANSFER_SRC"); break;
        case 7: snprintf(out, n, "TRANSFER_DST"); break;
        case 1000001002: snprintf(out, n, "PRESENT_SRC"); break;
        default: snprintf(out, n, "layout%d", l); break;
    }
}
static const char* LoadOpStr(int op) { return op == 0 ? "LOAD" : op == 1 ? "CLEAR" : op == 2 ? "DONT_CARE" : "NONE"; }
static const char* StoreOpStr(int op) { return op == 0 ? "STORE" : op == 1 ? "DONT_CARE" : "NONE"; }

static VKAPI_ATTR VkResult VKAPI_CALL SkyRT_CreateImage(VkDevice device, const VkImageCreateInfo* pCreateInfo,
                                                        const VkAllocationCallbacks* pAllocator, VkImage* pImage) {
    DeviceData* d = GetDev(Key(device));
    if (!d) return VK_ERROR_INITIALIZATION_FAILED;
    const VkImageUsageFlags u = pCreateInfo->usage;
    const bool isColorAtt = (u & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0;
    const bool isDepthAtt = (u & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0;
    const bool attachment = pCreateInfo->imageType == VK_IMAGE_TYPE_2D && pCreateInfo->extent.width >= 640 &&
                            pCreateInfo->extent.height >= 480 && (isColorAtt || isDepthAtt);

    // the main HDR color / D32 depth need STORAGE / SAMPLED so that our compute pass can use them later
    VkImageUsageFlags want = 0;
    if (attachment && d->rtEnabled && !g_noImgPatch && pCreateInfo->samples == VK_SAMPLE_COUNT_1_BIT &&
        !(u & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT)) {
        if (isColorAtt && pCreateInfo->format == VK_FORMAT_R16G16B16A16_SFLOAT)
            want = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        else if (isDepthAtt && pCreateInfo->format == VK_FORMAT_D32_SFLOAT)
            want = VK_IMAGE_USAGE_SAMPLED_BIT;
    }

    VkImageUsageFlags finalUsage = u;
    VkResult r = VK_ERROR_INITIALIZATION_FAILED;
    bool done = false;
    if (want && (u & want) != want) {
        VkImageCreateInfo ci = *pCreateInfo;
        ci.usage |= want;
        r = d->CreateImage(device, &ci, pAllocator, pImage);
        if (r == VK_SUCCESS) {
            finalUsage = ci.usage;
            d->imgPatched++;
            done = true;
        } else {
            d->imgPatchFail++;
            Logf("!!! patched vkCreateImage failed (%d), falling back to the original", (int)r);
        }
    }
    if (!done) r = d->CreateImage(device, pCreateInfo, pAllocator, pImage);

    if (r == VK_SUCCESS && !attachment) {
        std::lock_guard<std::mutex> lk(d->regMtx);
        d->imgs.erase(*pImage);  // the handle may be a recycled one
    }
    if (r == VK_SUCCESS && attachment) {
        {
            std::lock_guard<std::mutex> lk(d->regMtx);
            d->imgs[*pImage] = DeviceData::ImgInfo{pCreateInfo->format, pCreateInfo->extent.width,
                                                   pCreateInfo->extent.height, u, finalUsage};
        }
        if (d->imgLogged.fetch_add(1) < 120)
            Logf("attachment image %p %ux%u fmt=%s mips=%u layers=%u samples=%u usage=0x%x [%s%s%s%s]%s", (void*)*pImage,
                 pCreateInfo->extent.width, pCreateInfo->extent.height, FormatName((uint32_t)pCreateInfo->format),
                 pCreateInfo->mipLevels, pCreateInfo->arrayLayers, (unsigned)pCreateInfo->samples, (unsigned)finalUsage,
                 (finalUsage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) ? "COLOR_ATT " : "",
                 (finalUsage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ? "DEPTH_ATT " : "",
                 (finalUsage & VK_IMAGE_USAGE_SAMPLED_BIT) ? "SAMPLED " : "",
                 (finalUsage & VK_IMAGE_USAGE_STORAGE_BIT) ? "STORAGE" : "",
                 finalUsage != u ? " (patched: +STORAGE/+SAMPLED)" : "");
    }
    return r;
}

template <class AttDesc>
static void LogRP(DeviceData* d, const char* api, VkRenderPass rp, const AttDesc* atts, uint32_t n, uint32_t subpasses) {
    {
        std::vector<DeviceData::AttInfo> v;
        for (uint32_t i = 0; i < n; ++i)
            v.push_back(DeviceData::AttInfo{(uint32_t)atts[i].format, (uint32_t)atts[i].loadOp, (uint32_t)atts[i].storeOp,
                                            (uint32_t)atts[i].initialLayout, (uint32_t)atts[i].finalLayout});
        std::lock_guard<std::mutex> lk(d->regMtx);
        d->rps[rp] = std::move(v);
    }
    bool interesting = false;
    for (uint32_t i = 0; i < n; ++i)
        if (atts[i].format == VK_FORMAT_D32_SFLOAT || atts[i].format == VK_FORMAT_R16G16B16A16_SFLOAT) interesting = true;
    if (!interesting || d->rpLogged.fetch_add(1) >= 40) return;
    Logf("render pass (%s) %p: %u attachments, %u subpasses", api, (void*)rp, n, subpasses);
    for (uint32_t i = 0; i < n; ++i) {
        char a[40], b[40];
        LayoutStr((int)atts[i].initialLayout, a, sizeof a);
        LayoutStr((int)atts[i].finalLayout, b, sizeof b);
        Logf("  att[%u] %s samples=%u loadOp=%s storeOp=%s stencil(load=%s store=%s) %s -> %s", i,
             FormatName((uint32_t)atts[i].format), (unsigned)atts[i].samples, LoadOpStr((int)atts[i].loadOp),
             StoreOpStr((int)atts[i].storeOp), LoadOpStr((int)atts[i].stencilLoadOp),
             StoreOpStr((int)atts[i].stencilStoreOp), a, b);
    }
}

static VKAPI_ATTR VkResult VKAPI_CALL SkyRT_CreateRenderPass(VkDevice device, const VkRenderPassCreateInfo* pCreateInfo,
                                                             const VkAllocationCallbacks* pAllocator,
                                                             VkRenderPass* pRenderPass) {
    DeviceData* d = GetDev(Key(device));
    if (!d) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = d->CreateRenderPass(device, pCreateInfo, pAllocator, pRenderPass);
    if (r == VK_SUCCESS)
        LogRP(d, "v1", *pRenderPass, pCreateInfo->pAttachments, pCreateInfo->attachmentCount, pCreateInfo->subpassCount);
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL SkyRT_CreateRenderPass2(VkDevice device, const VkRenderPassCreateInfo2* pCreateInfo,
                                                              const VkAllocationCallbacks* pAllocator,
                                                              VkRenderPass* pRenderPass) {
    DeviceData* d = GetDev(Key(device));
    if (!d) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = d->CreateRenderPass2(device, pCreateInfo, pAllocator, pRenderPass);
    if (r == VK_SUCCESS)
        LogRP(d, "v2", *pRenderPass, pCreateInfo->pAttachments, pCreateInfo->attachmentCount, pCreateInfo->subpassCount);
    return r;
}

static void MainPassBegin(DeviceData* d, VkCommandBuffer cb, const VkRenderPassBeginInfo* pBegin);
static void MainPassEnd(DeviceData* d, VkCommandBuffer cb);

// ---- step 11.1 pass probe: Ctrl+Delete collects the render passes that draw, then hides them one by one (find the water pass)
struct PassProbe { VkRenderPass rp; uint32_t w, h; uint64_t draws; std::string desc; };
static std::mutex g_ppMtx;
static std::vector<PassProbe> g_ppList;                                                    // g_ppMtx
static std::unordered_map<VkCommandBuffer, std::pair<VkRenderPass, uint64_t>> g_cbPass;    // g_ppMtx: render pass + packed size of the open pass per cb
static std::atomic<int> g_ppState{0};   // 0 off, 1 collecting, 2 auto-cycle, 3 frozen
static std::atomic<int> g_ppHide{-1};
static void PassProbeBegin(VkCommandBuffer cb, const VkRenderPassBeginInfo* pb) {
    if (!g_ppState.load(std::memory_order_relaxed) || !pb) return;
    std::lock_guard<std::mutex> lk(g_ppMtx);
    g_cbPass[cb] = {pb->renderPass, ((uint64_t)pb->renderArea.extent.width << 32) | pb->renderArea.extent.height};
}
static bool PassProbeDraw(DeviceData* d, VkCommandBuffer cb) {  // true = skip this draw
    VkRenderPass rp = VK_NULL_HANDLE;
    uint64_t sz = 0;
    {
        std::lock_guard<std::mutex> lk(g_ppMtx);
        auto it = g_cbPass.find(cb);
        if (it == g_cbPass.end()) return false;
        rp = it->second.first;
        sz = it->second.second;
        for (PassProbe& e : g_ppList)
            if (e.rp == rp) { e.draws++; const int h = g_ppHide.load(); return h >= 0 && (size_t)h < g_ppList.size() && g_ppList[h].rp == rp; }
    }
    std::string desc;
    {
        std::lock_guard<std::mutex> lk(d->regMtx);
        auto rit = d->rps.find(rp);
        if (rit != d->rps.end()) {
            for (const DeviceData::AttInfo& a : rit->second) {
                char b[96];
                snprintf(b, sizeof b, "[%s %s] ", FormatName(a.format), LoadOpStr((int)a.loadOp));
                desc += b;
            }
        }
    }
    std::lock_guard<std::mutex> lk(g_ppMtx);
    for (PassProbe& e : g_ppList) if (e.rp == rp) return false;
    g_ppList.push_back(PassProbe{rp, (uint32_t)(sz >> 32), (uint32_t)(sz & 0xFFFFFFFFu), 1, desc});
    Logf("rt: PASSPROBE new pass #%zu: rp %p, %ux%u, attachments %s", g_ppList.size() - 1, (void*)rp, (uint32_t)(sz >> 32), (uint32_t)(sz & 0xFFFFFFFFu), desc.c_str());
    return false;
}
static void PassProbeLog(int h) {
    std::lock_guard<std::mutex> lk(g_ppMtx);
    if (h < 0 || (size_t)h >= g_ppList.size()) { Logf("rt: PASSPROBE nothing hidden"); return; }
    const PassProbe& e = g_ppList[h];
    Logf("rt: PASSPROBE HIDING pass #%d of %zu: rp %p, %ux%u, draws %llu, attachments %s", h, g_ppList.size(), (void*)e.rp, e.w, e.h,
         (unsigned long long)e.draws, e.desc.c_str());
}
static void PassProbeKey() {
    const int st = g_ppState.load();
    if (st == 0) { g_ppState = 1; Logf("rt: KEY Ctrl+Delete -> collecting render passes. Press again to start hiding them one by one"); }
    else if (st == 1) { g_ppState = 2; Logf("rt: PASSPROBE auto-cycle: all draws of one render pass are hidden at a time, every ~2 s. Press Ctrl+Delete when the WATER changes"); }
    else if (st == 2) { g_ppState = 3; Logf("rt: PASSPROBE FROZEN on the pass below:"); PassProbeLog(g_ppHide.load()); }
    else { g_ppState = 2; Logf("rt: PASSPROBE auto-cycle resumed"); }
}
static void PassProbeTick(uint64_t frame) {
    if (g_ppState.load() != 2 || frame % 150 != 0) return;
    size_t n;
    { std::lock_guard<std::mutex> lk(g_ppMtx); n = g_ppList.size(); }
    int h = g_ppHide.load() + 1;
    if ((size_t)h >= n) h = -1;
    g_ppHide = h;
    PassProbeLog(h);
}

static VKAPI_ATTR void VKAPI_CALL SkyRT_CmdBeginRenderPass(VkCommandBuffer cb, const VkRenderPassBeginInfo* pBegin,
                                                           VkSubpassContents contents) {
    DeviceData* d = GetDev(Key(cb));
    if (!d) return;
    d->rpBegin1.fetch_add(1, std::memory_order_relaxed);
    if (d->capturing.load(std::memory_order_relaxed) && pBegin) {
        std::lock_guard<std::mutex> lk(d->cmdMtx);
        const size_t n = d->cmdDraws[cb].size();
        d->passes.push_back(DeviceData::PassEv{pBegin->renderPass, pBegin->framebuffer, pBegin->renderArea.extent.width,
                                               pBegin->renderArea.extent.height, cb, n, n, true});
    }
    PassProbeBegin(cb, pBegin);
    d->CmdBeginRenderPass(cb, pBegin, contents);
    MainPassBegin(d, cb, pBegin);
}

static VKAPI_ATTR void VKAPI_CALL SkyRT_CmdEndRenderPass(VkCommandBuffer cb) {
    DeviceData* d = GetDev(Key(cb));
    if (!d) return;
    if (d->capturing.load(std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> lk(d->cmdMtx);
        for (auto it = d->passes.rbegin(); it != d->passes.rend(); ++it) {
            if (it->cb == cb && it->open) {
                it->drawEnd = d->cmdDraws[cb].size();
                it->open = false;
                break;
            }
        }
    }
    d->CmdEndRenderPass(cb);
    MainPassEnd(d, cb);
}

static VKAPI_ATTR VkResult VKAPI_CALL SkyRT_CreateImageView(VkDevice device, const VkImageViewCreateInfo* pCreateInfo,
                                                            const VkAllocationCallbacks* pAllocator, VkImageView* pView) {
    DeviceData* d = GetDev(Key(device));
    if (!d) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = d->CreateImageView(device, pCreateInfo, pAllocator, pView);
    if (r == VK_SUCCESS) {
        std::lock_guard<std::mutex> lk(d->regMtx);
        if (d->imgs.find(pCreateInfo->image) != d->imgs.end()) d->views[*pView] = DeviceData::ViewInfo{pCreateInfo->image};
        else d->views.erase(*pView);
    }
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL SkyRT_CreateFramebuffer(VkDevice device, const VkFramebufferCreateInfo* pCreateInfo,
                                                              const VkAllocationCallbacks* pAllocator,
                                                              VkFramebuffer* pFramebuffer) {
    DeviceData* d = GetDev(Key(device));
    if (!d) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = d->CreateFramebuffer(device, pCreateInfo, pAllocator, pFramebuffer);
    if (r == VK_SUCCESS && pCreateInfo->pAttachments && !(pCreateInfo->flags & VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT)) {
        std::lock_guard<std::mutex> lk(d->regMtx);
        DeviceData::FbInfo fi{pCreateInfo->renderPass, pCreateInfo->width, pCreateInfo->height, {}};
        bool any = false;
        for (uint32_t i = 0; i < pCreateInfo->attachmentCount; ++i) {
            fi.views.push_back(pCreateInfo->pAttachments[i]);
            if (d->views.find(pCreateInfo->pAttachments[i]) != d->views.end()) any = true;
        }
        if (any) d->fbs[*pFramebuffer] = std::move(fi);
        else d->fbs.erase(*pFramebuffer);
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL SkyRT_CmdBeginRenderPass2(VkCommandBuffer cb, const VkRenderPassBeginInfo* pBegin,
                                                            const VkSubpassBeginInfo* pSubpassBeginInfo) {
    DeviceData* d = GetDev(Key(cb));
    if (!d) return;
    d->rpBegin2.fetch_add(1, std::memory_order_relaxed);
    d->CmdBeginRenderPass2(cb, pBegin, pSubpassBeginInfo);
}

static VKAPI_ATTR void VKAPI_CALL SkyRT_GetDeviceQueue(VkDevice device, uint32_t family, uint32_t index, VkQueue* pQueue) {
    DeviceData* d = GetDev(Key(device));
    if (!d) return;
    d->GetDeviceQueue(device, family, index, pQueue);
    if (pQueue && *pQueue) {
        std::lock_guard<std::mutex> lk(d->regMtx);
        d->queueFamily[*pQueue] = family;
    }
}

static VKAPI_ATTR void VKAPI_CALL SkyRT_GetDeviceQueue2(VkDevice device, const VkDeviceQueueInfo2* pInfo, VkQueue* pQueue) {
    DeviceData* d = GetDev(Key(device));
    if (!d) return;
    d->GetDeviceQueue2(device, pInfo, pQueue);
    if (pInfo && pQueue && *pQueue) {
        std::lock_guard<std::mutex> lk(d->regMtx);
        d->queueFamily[*pQueue] = pInfo->queueFamilyIndex;
    }
}

// ------------------------------------------------------------ BLAS smoke test (runs once, from the first useful capture)
struct GpuBuf {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceAddress addr = 0;
};

static uint32_t FindMemType(DeviceData* d, uint32_t typeBits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp{};
    d->inst->GetPhysicalDeviceMemoryProperties(d->phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return 0xFFFFFFFFu;
}

// device-local buffer with its own memory (allocated with the device-address flag)
static bool MakeBuf(DeviceData* d, VkDeviceSize size, VkBufferUsageFlags usage, GpuBuf& out) {
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size;
    ci.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (d->CreateBuffer(d->device, &ci, nullptr, &out.buf) != VK_SUCCESS) return false;
    VkMemoryRequirements mr{};
    d->GetBufferMemoryRequirements(d->device, out.buf, &mr);
    const uint32_t type = FindMemType(d, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == 0xFFFFFFFFu) return false;
    VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &fi;
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = type;
    if (d->AllocateMemory(d->device, &ai, nullptr, &out.mem) != VK_SUCCESS) return false;
    if (d->BindBufferMemory(d->device, out.buf, out.mem, 0) != VK_SUCCESS) return false;
    VkBufferDeviceAddressInfo bi{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bi.buffer = out.buf;
    out.addr = d->GetBufferDeviceAddress(d->device, &bi);
    return out.addr != 0;
}

static void FreeBuf(DeviceData* d, GpuBuf& b) {
    if (b.buf) d->DestroyBuffer(d->device, b.buf, nullptr);
    if (b.mem) d->FreeMemory(d->device, b.mem, nullptr);
    b = GpuBuf();
}

static bool BufInfoOf(DeviceData* d, VkBuffer b, DeviceData::BufInfo& out) {
    std::lock_guard<std::mutex> lk(d->regMtx);
    auto it = d->bufs.find(b);
    if (it == d->bufs.end() || !it->second.mem) return false;
    out = it->second;
    return true;
}

static void BuildWorldBlasTest(DeviceData* d, const std::vector<DeviceData::Resolved>& res) {
    if (!d->rtEnabled || d->blasDone.load() || EnvIs1("SKYRT_NO_AS")) return;
    g_stage = "BuildWorldBlasTest";
    VkQueue queue = d->presentQueue.load();
    uint32_t family = 0xFFFFFFFFu;
    {
        std::lock_guard<std::mutex> lk(d->regMtx);
        auto it = d->queueFamily.find(queue);
        if (it != d->queueFamily.end()) family = it->second;
    }
    if (!queue || family == 0xFFFFFFFFu) { Logf("BLAS test skipped: present queue / its family is unknown"); return; }
    if (!d->setLoaderData || !d->CmdBuildAccelerationStructuresKHR || !d->GetAccelerationStructureBuildSizesKHR ||
        !d->CreateAccelerationStructureKHR || !d->GetBufferDeviceAddress) {
        Logf("BLAS test skipped: some required functions are missing");
        return;
    }

    // ---- 1. geometry from world-space indirect draws: position float3 at loc0, 16-bit indices
    struct Fmt { bool ok; uint32_t binding, stride, offset; };
    std::unordered_map<VkPipeline, Fmt> fmts;
    std::unordered_map<VkBuffer, VkDeviceAddress> addrs;
    auto addrOf = [&](VkBuffer b) -> VkDeviceAddress {
        auto it = addrs.find(b);
        if (it != addrs.end()) return it->second;
        VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        ai.buffer = b;
        const VkDeviceAddress a = d->GetBufferDeviceAddress(d->device, &ai);
        addrs[b] = a;
        return a;
    };

    std::vector<VkAccelerationStructureGeometryKHR> geoms;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
    uint64_t tris = 0, skipped = 0, candidates = 0;
    for (const DeviceData::Resolved& r : res) {
        if (!r.fromIndirect) continue;
        candidates++;
        if (geoms.size() >= 4096) { skipped++; continue; }
        if (r.st.ibType != VK_INDEX_TYPE_UINT16 || !r.st.ib || !r.st.pipe) { skipped++; continue; }
        auto fit = fmts.find(r.st.pipe);
        if (fit == fmts.end()) {
            Fmt f{false, 0, 0, 0};
            {
                std::lock_guard<std::mutex> lk(d->regMtx);
                auto pit = d->pipes.find(r.st.pipe);
                if (pit != d->pipes.end()) {
                    for (const DeviceData::PipeInfo::Attr& a : pit->second.attrs) {
                        if (a.loc != 0 || a.format != (uint32_t)VK_FORMAT_R32G32B32_SFLOAT) continue;
                        for (const DeviceData::PipeInfo::Bind& b : pit->second.binds)
                            if (b.binding == a.binding && b.rate == 0) f = Fmt{true, a.binding, b.stride, a.offset};
                    }
                }
            }
            fit = fmts.emplace(r.st.pipe, f).first;
        }
        const Fmt f = fit->second;
        if (!f.ok || f.binding >= 8 || !r.st.vb[f.binding] || f.stride < 12 || r.c.indexCount < 3 ||
            r.c.vertexOffset < 0) { skipped++; continue; }
        DeviceData::BufInfo vbi{}, ibi{};
        if (!BufInfoOf(d, r.st.vb[f.binding], vbi) || !BufInfoOf(d, r.st.ib, ibi)) { skipped++; continue; }
        const uint64_t vbase = r.st.vbOff[f.binding] + f.offset + (uint64_t)r.c.vertexOffset * f.stride;
        const uint64_t ibase = r.st.ibOff + (uint64_t)r.c.firstIndex * 2;
        if (vbase + f.stride > vbi.size || ibase + (uint64_t)r.c.indexCount * 2 > ibi.size) { skipped++; continue; }
        const VkDeviceAddress va = addrOf(r.st.vb[f.binding]);
        const VkDeviceAddress ia = addrOf(r.st.ib);
        if (!va || !ia) { skipped++; continue; }
        const uint64_t avail = (vbi.size - vbase) / f.stride;  // vertices available from vbase

        VkAccelerationStructureGeometryKHR g{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        g.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        g.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        g.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        g.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        g.geometry.triangles.vertexData.deviceAddress = va + vbase;
        g.geometry.triangles.vertexStride = f.stride;
        g.geometry.triangles.maxVertex = (uint32_t)(avail ? avail - 1 : 0);
        g.geometry.triangles.indexType = VK_INDEX_TYPE_UINT16;
        g.geometry.triangles.indexData.deviceAddress = ia + ibase;
        VkAccelerationStructureBuildRangeInfoKHR rg{};
        rg.primitiveCount = r.c.indexCount / 3;
        geoms.push_back(g);
        ranges.push_back(rg);
        tris += rg.primitiveCount;
    }
    if (geoms.empty()) {
        Logf("BLAS test: no suitable draws in this capture (candidates %llu, skipped %llu), will retry at the next capture",
             (unsigned long long)candidates, (unsigned long long)skipped);
        return;
    }
    d->blasDone = true;
    Logf("BLAS test: %zu geometries, %llu triangles (indirect draws seen %llu, skipped %llu)", geoms.size(),
         (unsigned long long)tris, (unsigned long long)candidates, (unsigned long long)skipped);

    // ---- 2. build sizes
    VkAccelerationStructureBuildGeometryInfoKHR bi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    bi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bi.geometryCount = (uint32_t)geoms.size();
    bi.pGeometries = geoms.data();
    std::vector<uint32_t> maxPrim(geoms.size());
    for (size_t i = 0; i < geoms.size(); ++i) maxPrim[i] = ranges[i].primitiveCount;
    VkAccelerationStructureBuildSizesInfoKHR sz{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    d->GetAccelerationStructureBuildSizesKHR(d->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi,
                                             maxPrim.data(), &sz);

    VkDeviceSize scratchAlign = 256;
    if (d->inst->GetPhysicalDeviceProperties2) {
        VkPhysicalDeviceAccelerationStructurePropertiesKHR asp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &asp;
        d->inst->GetPhysicalDeviceProperties2(d->phys, &p2);
        if (asp.minAccelerationStructureScratchOffsetAlignment) scratchAlign = asp.minAccelerationStructureScratchOffsetAlignment;
        Logf("AS limits: maxGeometryCount=%llu maxPrimitiveCount=%llu scratchAlign=%llu",
             (unsigned long long)asp.maxGeometryCount, (unsigned long long)asp.maxPrimitiveCount,
             (unsigned long long)scratchAlign);
    }
    Logf("BLAS test: AS size %.2f MB, scratch %.2f MB", sz.accelerationStructureSize / 1048576.0,
         sz.buildScratchSize / 1048576.0);

    // ---- 3. objects
    GpuBuf asBuf, scratch;
    VkAccelerationStructureKHR as = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    const char* failed = nullptr;
    if (!MakeBuf(d, (sz.accelerationStructureSize + 255) & ~255ull, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, asBuf))
        failed = "AS storage buffer";
    else if (!MakeBuf(d, sz.buildScratchSize + scratchAlign, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, scratch))
        failed = "scratch buffer";
    if (!failed) {
        VkAccelerationStructureCreateInfoKHR aci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        aci.buffer = asBuf.buf;
        aci.offset = 0;
        aci.size = sz.accelerationStructureSize;
        aci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        if (d->CreateAccelerationStructureKHR(d->device, &aci, nullptr, &as) != VK_SUCCESS) failed = "vkCreateAccelerationStructureKHR";
    }
    if (!failed) {
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pci.queueFamilyIndex = family;
        if (d->CreateCommandPool(d->device, &pci, nullptr, &pool) != VK_SUCCESS) failed = "command pool";
    }
    if (!failed) {
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        if (d->AllocateCommandBuffers(d->device, &cai, &cb) != VK_SUCCESS || d->setLoaderData(d->device, cb) != VK_SUCCESS)
            failed = "command buffer";
    }
    if (!failed) {
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (d->CreateFence(d->device, &fci, nullptr, &fence) != VK_SUCCESS) failed = "fence";
    }
    if (failed) {
        Logf("!!! BLAS test: failed to create %s (resources are leaked on purpose, this runs once)", failed);
        return;
    }

    // ---- 4. record + submit + wait
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    d->BeginCommandBuffer(cb, &cbi);
    bi.dstAccelerationStructure = as;
    bi.scratchData.deviceAddress = (scratch.addr + scratchAlign - 1) & ~(scratchAlign - 1);
    const VkAccelerationStructureBuildRangeInfoKHR* pRanges = ranges.data();
    d->CmdBuildAccelerationStructuresKHR(cb, 1, &bi, &pRanges);
    d->EndCommandBuffer(cb);

    VkSubmitInfo sub{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    sub.commandBufferCount = 1;
    sub.pCommandBuffers = &cb;
    LARGE_INTEGER t0, t1, fq;
    QueryPerformanceFrequency(&fq);
    QueryPerformanceCounter(&t0);
    VkResult sr = d->QueueSubmit(queue, 1, &sub, fence);
    VkResult wr = (sr == VK_SUCCESS) ? d->WaitForFences(d->device, 1, &fence, VK_TRUE, 5000000000ull) : sr;
    QueryPerformanceCounter(&t1);
    const double ms = (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)fq.QuadPart;
    Logf("BLAS test: submit=%d wait=%d, build took %.2f ms (includes waiting for the game's earlier GPU work)", (int)sr,
         (int)wr, ms);
    FlushToDisk();

    if (wr == VK_SUCCESS) {  // cleanup only when the GPU is done with everything
        d->DestroyFence(d->device, fence, nullptr);
        d->DestroyCommandPool(d->device, pool, nullptr);
        d->DestroyAccelerationStructureKHR(d->device, as, nullptr);
        FreeBuf(d, asBuf);
        FreeBuf(d, scratch);
        Logf("BLAS test: OK, objects destroyed");
    }
}

// ============================================================ step 5: compute pass after the main render pass
static VkImageView GetOwnView(DeviceData* d, VkImage img, VkFormat fmt, VkImageAspectFlags aspect) {
    // caller holds d->paintMtx
    auto it = d->ownViews.find(img);
    if (it != d->ownViews.end()) return it->second;
    VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    ci.image = img;
    ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ci.format = fmt;
    ci.subresourceRange.aspectMask = aspect;
    ci.subresourceRange.levelCount = 1;
    ci.subresourceRange.layerCount = 1;
    VkImageView v = VK_NULL_HANDLE;
    const VkResult r = d->CreateImageView(d->device, &ci, nullptr, &v);  // the original function: not tracked as a game view
    if (r != VK_SUCCESS) { Logf("!!! paint: vkCreateImageView failed (%d)", (int)r); return VK_NULL_HANDLE; }
    d->ownViews[img] = v;
    return v;
}

static bool MakeBufEx(DeviceData* d, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
                      VkBuffer& buf, VkDeviceMemory& mem, VkDeviceAddress& addr);

struct FxParams {
    float vp[16];
    float inv[16];
    float sun[4];
    float cam[4];
    uint32_t misc[4];
    uint32_t flags[4];
    float fx[4];
    float fx2[4];
    float pvp[16];   // step 20: previous frame view-projection
    float tp[4];     // x = history valid, y = max samples
    float lt[4];     // step 21: x = light strength, y = range, z = brightness threshold, w = GI multiplier (strength * 0.5, applied in the trace pass)
    uint32_t lm[4];  // x = tile size, y = tiles in x, z = shadow rays for lights, w = bit0 lights on, bit1 debug
};
static const uint32_t kLightMax = 256, kLightBufBytes = 16 + 256 * 32;

static VkPipeline MakeComputePipe(DeviceData* d, const uint32_t* code, size_t bytes, const char* what) {
    VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sm.codeSize = bytes;
    sm.pCode = code;
    VkShaderModule mod = VK_NULL_HANDLE;
    VkResult r = d->CreateShaderModule(d->device, &sm, nullptr, &mod);
    if (r != VK_SUCCESS) { Logf("!!! rt: shader module (%s) failed (%d)", what, (int)r); return VK_NULL_HANDLE; }
    VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp.stage.module = mod;
    cp.stage.pName = "main";
    cp.layout = d->paintPl;
    VkPipeline pipe = VK_NULL_HANDLE;
    r = d->CreateComputePipelines(d->device, VK_NULL_HANDLE, 1, &cp, nullptr, &pipe);
    d->DestroyShaderModule(d->device, mod, nullptr);
    if (r != VK_SUCCESS) { Logf("!!! rt: compute pipeline (%s) failed (%d)", what, (int)r); return VK_NULL_HANDLE; }
    return pipe;
}

static bool EnsurePaintResources(DeviceData* d) {
    // caller holds d->paintMtx. (the names say "paint" for historical reasons: these are the ray-query pipelines now)
    if (d->paintTried) return d->paintOk;
    d->paintTried = true;
    g_stage = "EnsurePaintResources";

    VkDescriptorSetLayoutBinding b[11] = {};
    const VkDescriptorType types[11] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                       VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                       VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                       VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
    for (uint32_t i = 0; i < 11; ++i) {
        b[i].binding = i;
        b[i].descriptorType = types[i];
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dl.bindingCount = 11;
    dl.pBindings = b;
    VkResult r = d->CreateDescriptorSetLayout(d->device, &dl, nullptr, &d->paintDsl);
    if (r != VK_SUCCESS) { Logf("!!! rt: descriptor set layout failed (%d)", (int)r); return false; }

    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &d->paintDsl;
    r = d->CreatePipelineLayout(d->device, &pl, nullptr, &d->paintPl);
    if (r != VK_SUCCESS) { Logf("!!! rt: pipeline layout failed (%d)", (int)r); return false; }

    d->paintPipe = MakeComputePipe(d, kRtSpv, sizeof(kRtSpv), "trace");
    d->fxPipe = MakeComputePipe(d, kRtFxSpv, sizeof(kRtFxSpv), "composite");
    if (!d->paintPipe || !d->fxPipe) return false;
    d->lightPipe = MakeComputePipe(d, kRtLightSpv, sizeof(kRtLightSpv), "lights");   // optional: without it the emissive light is simply off

    VkDescriptorPoolSize ps[5] = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 16 * 7}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 16},
                                  {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 16}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16},
                                  {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16}};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.maxSets = 16;
    pi.poolSizeCount = 5;
    pi.pPoolSizes = ps;
    r = d->CreateDescriptorPool(d->device, &pi, nullptr, &d->paintPool);
    if (r != VK_SUCCESS) { Logf("!!! rt: descriptor pool failed (%d)", (int)r); return false; }
    VkDescriptorSetLayout layouts[16];
    for (int i = 0; i < 16; ++i) layouts[i] = d->paintDsl;
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = d->paintPool;
    ai.descriptorSetCount = 16;
    ai.pSetLayouts = layouts;
    r = d->AllocateDescriptorSets(d->device, &ai, d->paintSets);
    if (r != VK_SUCCESS) { Logf("!!! rt: descriptor sets failed (%d)", (int)r); return false; }

    {   // step 21: the light list (device local) and a tiny host-visible buffer the header is copied to for the log
        VkDeviceAddress da = 0;
        if (!d->lightPipe || !MakeBufEx(d, kLightBufBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, d->lightBuf, d->lightMem, da)) {
            Logf("!!! rt: light list buffer / pipeline not available - emissive light disabled");
            d->lightBuf = VK_NULL_HANDLE;
        } else if (MakeBufEx(d, 64, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, d->statBuf, d->statMem, da)) {
            void* sp = nullptr;
            if (d->MapMemory(d->device, d->statMem, 0, VK_WHOLE_SIZE, 0, &sp) == VK_SUCCESS && sp) { d->statPtr = (char*)sp; memset(sp, 0, 64); }
        }
    }
    // one small host-visible parameter buffer per descriptor set, bound once
    for (int i = 0; i < 16; ++i) {
        VkDeviceAddress dummy = 0;
        if (!MakeBufEx(d, 512, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       d->parBuf[i], d->parMem[i], dummy)) { Logf("!!! rt: parameter buffer %d failed", i); return false; }
        void* mp = nullptr;
        if (d->MapMemory(d->device, d->parMem[i], 0, VK_WHOLE_SIZE, 0, &mp) != VK_SUCCESS || !mp) { Logf("!!! rt: parameter buffer map failed"); return false; }
        d->parPtr[i] = (char*)mp;
        memset(mp, 0, 512);
        VkDescriptorBufferInfo bi{d->parBuf[i], 0, sizeof(FxParams)};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = d->paintSets[i];
        w.dstBinding = 5;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w.pBufferInfo = &bi;
        d->UpdateDescriptorSets(d->device, 1, &w, 0, nullptr);
        if (d->lightBuf) {   // binding 10: the light list
            VkDescriptorBufferInfo lbi{d->lightBuf, 0, kLightBufBytes};
            VkWriteDescriptorSet lw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            lw.dstSet = d->paintSets[i];
            lw.dstBinding = 10;
            lw.descriptorCount = 1;
            lw.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            lw.pBufferInfo = &lbi;
            d->UpdateDescriptorSets(d->device, 1, &lw, 0, nullptr);
        }
    }

    d->paintOk = true;
    Logf("rt: trace + composite%s pipelines ready (16 descriptor sets in a ring, params %zu bytes)", d->lightBuf ? " + lights" : "", sizeof(FxParams));
    return true;
}

// result images of the trace pass (same size as the main color image); caller holds d->paintMtx
static bool EnsureFxImages(DeviceData* d, uint32_t w, uint32_t h) {
    if (d->giImg && d->fxW == w && d->fxH == h) return true;
    if (d->giImg) Logf("rt: resolution changed to %ux%u - allocating new result images (old ones are leaked on purpose)", w, h);
    auto make = [&](VkFormat fmt, VkImage& img, VkDeviceMemory& mem, VkImageView& view) -> bool {
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = fmt;
        ci.extent = {w, h, 1};
        ci.mipLevels = 1;
        ci.arrayLayers = 1;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.usage = VK_IMAGE_USAGE_STORAGE_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (d->CreateImage(d->device, &ci, nullptr, &img) != VK_SUCCESS) return false;
        VkMemoryRequirements mr{};
        d->GetImageMemoryRequirements(d->device, img, &mr);
        const uint32_t type = FindMemType(d, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (type == 0xFFFFFFFFu) return false;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = type;
        if (d->AllocateMemory(d->device, &ai, nullptr, &mem) != VK_SUCCESS) return false;
        if (d->BindImageMemory(d->device, img, mem, 0) != VK_SUCCESS) return false;
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = img;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = fmt;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        return d->CreateImageView(d->device, &vi, nullptr, &view) == VK_SUCCESS;
    };
    VkImage gi = VK_NULL_HANDLE, sh = VK_NULL_HANDLE;
    VkDeviceMemory gm = VK_NULL_HANDLE, sm = VK_NULL_HANDLE;
    VkImageView gv = VK_NULL_HANDLE, sv = VK_NULL_HANDLE;
    if (!make(VK_FORMAT_R16G16B16A16_SFLOAT, gi, gm, gv) || !make(VK_FORMAT_R32_SFLOAT, sh, sm, sv)) {
        Logf("!!! rt: could not create the result images (%ux%u)", w, h);
        return false;
    }
    {   // step 20: history images
        const VkFormat hf[4] = {VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT};
        VkImage hi[4] = {}; VkDeviceMemory hm[4] = {}; VkImageView hv[4] = {};
        bool hok = true;
        for (int i = 0; i < 4 && hok; ++i) hok = make(hf[i], hi[i], hm[i], hv[i]);
        if (!hok) { Logf("!!! rt: could not create the temporal history images - temporal accumulation disabled"); for (int i = 0; i < 4; ++i) d->hView[i] = VK_NULL_HANDLE; }
        else for (int i = 0; i < 4; ++i) { d->hImg[i] = hi[i]; d->hMem[i] = hm[i]; d->hView[i] = hv[i]; }
        d->histValid = false;
        d->histCur = 0;
    }
    d->giImg = gi; d->giMem = gm; d->giView = gv;
    d->shImg = sh; d->shMem = sm; d->shView = sv;
    d->fxW = w;
    d->fxH = h;
    Logf("rt: result images ready (%ux%u: RGBA16F bounce+AO, R32F shadow)", w, h);
    return true;
}

// ---------------------------------------------------------------- per-frame acceleration structures
static bool MakeBufEx(DeviceData* d, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
                      VkBuffer& buf, VkDeviceMemory& mem, VkDeviceAddress& addr) {
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size;
    ci.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (d->CreateBuffer(d->device, &ci, nullptr, &buf) != VK_SUCCESS) return false;
    VkMemoryRequirements mr{};
    d->GetBufferMemoryRequirements(d->device, buf, &mr);
    const uint32_t type = FindMemType(d, mr.memoryTypeBits, props);
    if (type == 0xFFFFFFFFu) return false;
    VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &fi;
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = type;
    if (d->AllocateMemory(d->device, &ai, nullptr, &mem) != VK_SUCCESS) return false;
    if (d->BindBufferMemory(d->device, buf, mem, 0) != VK_SUCCESS) return false;
    VkBufferDeviceAddressInfo bi{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bi.buffer = buf;
    addr = d->GetBufferDeviceAddress(d->device, &bi);
    return addr != 0;
}

static const VkDeviceSize kBlasCap = 48ull << 20, kScratchCap = 40ull << 20;

// caller holds d->paintMtx
static bool EnsureRtSlots(DeviceData* d) {
    if (d->rtSlotsTried) return d->rtSlotsOk;
    d->rtSlotsTried = true;
    g_stage = "EnsureRtSlots";
    if (!d->GetAccelerationStructureDeviceAddressKHR || !d->CmdBuildAccelerationStructuresKHR ||
        !d->CreateAccelerationStructureKHR || !d->GetAccelerationStructureBuildSizesKHR || !d->MapMemory) {
        Logf("!!! rt: acceleration structure functions are missing - no per-frame AS");
        return false;
    }
    if (d->inst->GetPhysicalDeviceProperties2) {
        VkPhysicalDeviceAccelerationStructurePropertiesKHR asp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &asp;
        d->inst->GetPhysicalDeviceProperties2(d->phys, &p2);
        if (asp.minAccelerationStructureScratchOffsetAlignment) d->asScratchAlign = asp.minAccelerationStructureScratchOffsetAlignment;
    }
    // size of a TLAS with one instance
    VkAccelerationStructureGeometryKHR tg{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    tg.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tg.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    VkAccelerationStructureBuildGeometryInfoKHR tb{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    tb.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    tb.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    tb.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    tb.geometryCount = 1;
    tb.pGeometries = &tg;
    const uint32_t one = 1;
    VkAccelerationStructureBuildSizesInfoKHR tsz{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    d->GetAccelerationStructureBuildSizesKHR(d->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tb, &one, &tsz);
    const VkDeviceSize tlasCap = (tsz.accelerationStructureSize + 255) & ~255ull;

    for (int i = 0; i < 8; ++i) {
        DeviceData::RtSlot& sl = d->rtSlots[i];
        VkDeviceAddress dummy = 0;
        const char* failed = nullptr;
        if (!MakeBufEx(d, kBlasCap, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       sl.blasBuf, sl.blasMem, dummy)) failed = "BLAS storage";
        else if (!MakeBufEx(d, tlasCap, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            sl.tlasBuf, sl.tlasMem, dummy)) failed = "TLAS storage";
        else if (!MakeBufEx(d, kScratchCap + d->asScratchAlign, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            sl.scratchBuf, sl.scratchMem, sl.scratchAddr)) failed = "scratch";
        else if (!MakeBufEx(d, 4096, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            sl.instBuf, sl.instMem, sl.instAddr)) failed = "instance buffer";
        if (!failed) {
            VkAccelerationStructureCreateInfoKHR aci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
            aci.buffer = sl.blasBuf;
            aci.size = kBlasCap;
            aci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            if (d->CreateAccelerationStructureKHR(d->device, &aci, nullptr, &sl.blas) != VK_SUCCESS) failed = "BLAS object";
            else {
                aci.buffer = sl.tlasBuf;
                aci.size = tlasCap;
                aci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
                if (d->CreateAccelerationStructureKHR(d->device, &aci, nullptr, &sl.tlas) != VK_SUCCESS) failed = "TLAS object";
            }
        }
        if (!failed) {
            VkAccelerationStructureDeviceAddressInfoKHR ai{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
            ai.accelerationStructure = sl.blas;
            sl.blasAddr = d->GetAccelerationStructureDeviceAddressKHR(d->device, &ai);
            void* mp = nullptr;
            if (!sl.blasAddr || d->MapMemory(d->device, sl.instMem, 0, VK_WHOLE_SIZE, 0, &mp) != VK_SUCCESS || !mp) failed = "map/address";
            else {
                VkAccelerationStructureInstanceKHR inst{};
                inst.transform.matrix[0][0] = inst.transform.matrix[1][1] = inst.transform.matrix[2][2] = 1.0f;
                inst.mask = 0xFF;
                inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
                inst.accelerationStructureReference = sl.blasAddr;
                memcpy(mp, &inst, sizeof inst);
            }
        }
        if (failed) { Logf("!!! rt: slot %d: failed to create %s - no per-frame AS", i, failed); return false; }
        sl.blasCap = kBlasCap;
        sl.tlasCap = tlasCap;
        sl.scratchCap = kScratchCap;
        sl.ok = true;
    }
    d->rtSlotsOk = true;
    Logf("rt: 8 AS slots ready (BLAS %llu MB, scratch %llu MB, TLAS %llu KB each; scratch align %llu)",
         (unsigned long long)(kBlasCap >> 20), (unsigned long long)(kScratchCap >> 20), (unsigned long long)(tlasCap >> 10),
         (unsigned long long)d->asScratchAlign);
    return true;
}

// ---------------------------------------------------------------- geometry of the main pass, collected while recording
// ---- step 9 diagnostic: which blended (non-opaque) pipeline is the water / ice? Ctrl+PageDown hides them one by one.
struct ProbePipe { VkPipeline pipe; uint32_t vstride, attrs; bool instanced; uint64_t draws, maxIdx, tris; bool opaque; uint64_t vs, fs; bool depthTest, depthWrite; uint32_t src, dst; bool nonIndexed; };
static std::mutex g_probeMtx;
static std::vector<ProbePipe> g_probeList;      // g_probeMtx: blended pipelines in order of first appearance
static std::atomic<bool> g_probeOn{false};      // set by the first Ctrl+PageDown
static std::atomic<int> g_hide{-1};             // index into g_probeList of the pipeline whose draws are skipped
static void ProbeBlended(DeviceData* d, const DeviceData::CmdState& st, const std::vector<VkDrawIndexedIndirectCommand>& cmds, bool nonIndexed = false) {
    if (!g_probeOn.load(std::memory_order_relaxed)) return;
    DeviceData::PipeInfo pi{};
    {
        std::lock_guard<std::mutex> lk(d->regMtx);
        auto pit = d->pipes.find(st.pipe);
        if (pit == d->pipes.end()) return;
        pi = pit->second;
    }
    uint64_t tris = 0, mx = 0;
    for (const VkDrawIndexedIndirectCommand& c : cmds) { tris += (uint64_t)(c.indexCount / 3) * std::max(1u, c.instanceCount); mx = std::max<uint64_t>(mx, c.indexCount); }
    std::lock_guard<std::mutex> lk(g_probeMtx);
    for (ProbePipe& e : g_probeList)
        if (e.pipe == st.pipe) { e.draws += cmds.size(); e.tris += tris; e.maxIdx = std::max(e.maxIdx, mx); return; }
    if (g_probeList.size() >= 400) return;
    ProbePipe e{st.pipe, pi.binds.empty() ? 0u : pi.binds[0].stride, (uint32_t)pi.attrs.size(), pi.instanced, cmds.size(), mx, tris, pi.opaque, pi.vsHash, pi.fsHash, pi.depthTest, pi.depthWrite, pi.srcColor, pi.dstColor, nonIndexed};
    g_probeList.push_back(e);
    Logf("rt: PROBE new pipeline #%zu%s: pipe %p, %s, vstride %u, attrs %u, instanced %d, first draw: indexCount max %llu | vs %016llx fs %016llx depthTest %d depthWrite %d blend %u->%u",
         g_probeList.size() - 1, e.nonIndexed ? " (NON-INDEXED vkCmdDraw)" : "", (void*)e.pipe, e.opaque ? "opaque (in RT scene)" : "blended (not in RT scene)", e.vstride, e.attrs, e.instanced ? 1 : 0, (unsigned long long)mx,
         (unsigned long long)e.vs, (unsigned long long)e.fs, e.depthTest ? 1 : 0, e.depthWrite ? 1 : 0, e.src, e.dst);
}
static std::atomic<int> g_hideGroup{0};         // step 12: 0 off, 1 all blended, 2 blended stride 28, 3 blended stride 12, 4 blended stride 32/36
static const char* kGroupNames[5] = {"off", "ALL blended pipelines", "blended, vertex stride 28 (ground layers?)", "blended, vertex stride 12", "blended, vertex stride 32/36"};
static bool ShouldHide(DeviceData* d, VkCommandBuffer cb) {
    const int h = g_hide.load(std::memory_order_relaxed);
    const int grp = g_hideGroup.load(std::memory_order_relaxed);
    if (h < 0 && grp <= 0) return false;
    {
        std::lock_guard<std::mutex> lk(d->paintMtx);
        if (!d->mainCb.count(cb)) return false;
    }
    VkPipeline cur = VK_NULL_HANDLE;
    {
        std::lock_guard<std::mutex> lk(d->cmdMtx);
        auto it = d->cmdState.find(cb);
        if (it == d->cmdState.end()) return false;
        cur = it->second.pipe;
    }
    if (grp > 0) {
        std::lock_guard<std::mutex> lk(d->regMtx);
        auto pit = d->pipes.find(cur);
        if (pit != d->pipes.end() && !pit->second.opaque) {
            const uint32_t stv = pit->second.binds.empty() ? 0u : pit->second.binds[0].stride;
            if (grp == 1) return true;
            if (grp == 2 && stv == 28) return true;
            if (grp == 3 && stv == 12) return true;
            if (grp == 4 && (stv == 32 || stv == 36)) return true;
        }
    }
    if (h < 0) return false;
    VkPipeline target = VK_NULL_HANDLE;
    {
        std::lock_guard<std::mutex> lk(g_probeMtx);
        if ((size_t)h >= g_probeList.size()) return false;
        target = g_probeList[h].pipe;
    }
    return cur == target;
}
static std::atomic<int> g_probeState{0};        // 0 off, 1 collecting, 2 auto-cycling (hides the next pipeline every ~2 s), 3 frozen on one
static void ProbeLogHide(int h) {
    std::lock_guard<std::mutex> lk(g_probeMtx);
    if (h < 0 || (size_t)h >= g_probeList.size()) { Logf("rt: PROBE nothing hidden"); return; }
    const ProbePipe& e = g_probeList[h];
    Logf("rt: PROBE HIDING pipeline #%d of %zu%s: pipe %p, %s, vstride %u, attrs %u, instanced %d, draws %llu, tris %llu, max indexCount %llu | vs %016llx fs %016llx depthTest %d depthWrite %d blend %u->%u",
         h, g_probeList.size(), e.nonIndexed ? " (NON-INDEXED vkCmdDraw)" : "", (void*)e.pipe, e.opaque ? "opaque" : "blended", e.vstride, e.attrs, e.instanced ? 1 : 0,
         (unsigned long long)e.draws, (unsigned long long)e.tris, (unsigned long long)e.maxIdx,
         (unsigned long long)e.vs, (unsigned long long)e.fs, e.depthTest ? 1 : 0, e.depthWrite ? 1 : 0, e.src, e.dst);
}
static void ProbeKey(bool pageUp) {  // Ctrl+PageDown: collect -> auto-cycle -> freeze <-> resume; Ctrl+PageUp: stop and show everything again
    if (pageUp) { g_probeState = 0; g_hide = -1; g_hideGroup = 0; Logf("rt: KEY Ctrl+PageUp -> probe stopped, everything visible"); return; }
    const int st = g_probeState.load();
    if (st == 0) { g_probeOn = true; g_probeState = 1; Logf("rt: KEY Ctrl+PageDown -> collecting pipelines (look around, water in view). Press again to start hiding them one by one"); }
    else if (st == 1) { g_probeState = 2; Logf("rt: PROBE auto-cycle: a new pipeline is hidden every ~2 s. Press Ctrl+PageDown the moment the WATER disappears (or changes)"); }
    else if (st == 2) { g_probeState = 3; Logf("rt: PROBE FROZEN on the pipeline below (this is the one you saw change):"); ProbeLogHide(g_hide.load()); }
    else { g_probeState = 2; Logf("rt: PROBE auto-cycle resumed"); }
}
static void ProbeTick(uint64_t frame) {
    if (g_probeState.load() != 2 || frame % 150 != 0) return;
    size_t n;
    { std::lock_guard<std::mutex> lk(g_probeMtx); n = g_probeList.size(); }
    int h = g_hide.load() + 1;
    if ((size_t)h >= n) h = -1;
    g_hide = h;
    ProbeLogHide(h);
}

// step 14: a full BLAS rebuild of the whole cache is a GPU hitch that grows with the cache; while flying, new terrain arrives all the time.
// Big cache -> rebuild rarely (the newest chunks are simply missing from the shadows for a moment), a large share of new geometry -> a bit sooner.
static uint32_t RebuildGap(uint64_t cacheTris, uint64_t lastTris) {
    uint64_t gap = std::max<uint64_t>(2, std::min<uint64_t>(150, cacheTris / 5000));
    const uint64_t diff = cacheTris > lastTris ? cacheTris - lastTris : lastTris - cacheTris;
    if (lastTris && diff * 100 > lastTris * 25) gap = std::max<uint64_t>(2, gap / 3);   // cache changed by >25 %
    return (uint32_t)gap;
}
// shader fingerprints of the water pipeline (found with the probe). 17a63a0f.. / 22efde0f.. = first blended pipeline of the probe list; the other candidate was 0bbf0e40.. / cdceac11.. (SkyRT.cfg: watervs= / waterfs=)
static uint64_t kWaterVs = 0x17a63a0f3447fc7cull, kWaterFs = 0x22efde0f73a797a4ull;
static void MainDraw(DeviceData* d, VkCommandBuffer cb, bool indirect, VkBuffer indBuf, VkDeviceSize indOff, uint32_t drawCount,
                     uint32_t stride, const VkDrawIndexedIndirectCommand* direct) {
    {
        std::lock_guard<std::mutex> lk(d->paintMtx);
        if (!d->mainCb.count(cb)) return;
    }
    struct Timer { DeviceData* d; LARGE_INTEGER t0, fq; Timer(DeviceData* dd) : d(dd) { QueryPerformanceFrequency(&fq); QueryPerformanceCounter(&t0); }
                   ~Timer() { LARGE_INTEGER t1; QueryPerformanceCounter(&t1); d->usDraw += (uint64_t)((t1.QuadPart - t0.QuadPart) * 1000000ll / fq.QuadPart); } } timer(d);
    DeviceData::CmdState st{};
    {
        std::lock_guard<std::mutex> lk(d->cmdMtx);
        auto it = d->cmdState.find(cb);
        if (it == d->cmdState.end()) return;
        st = it->second;
    }
    std::vector<VkDrawIndexedIndirectCommand> cmds;
    bool unreadable = false;
    if (indirect) {
        if (!ReadIndirect(d, indBuf, indOff, drawCount, stride ? stride : (uint32_t)sizeof(VkDrawIndexedIndirectCommand), cmds))
            unreadable = true;
    } else {
        cmds.push_back(*direct);
    }

    ProbeBlended(d, st, cmds);
    {   // step 15: print the full vertex layout of every pipeline the first time the main pass uses it (skinned characters have joint indices / weights / instance data)
        static std::mutex lm;
        static std::unordered_set<VkPipeline> logged;
        static int count = 0;
        std::lock_guard<std::mutex> ll(lm);
        if (count < 150 && logged.insert(st.pipe).second) {
            DeviceData::PipeInfo pi;
            bool have = false;
            { std::lock_guard<std::mutex> lk(d->regMtx); auto pit = d->pipes.find(st.pipe); if (pit != d->pipes.end()) { pi = pit->second; have = true; } }
            if (have) {
                ++count;
                uint32_t ic = 0, inst = 0;
                for (const VkDrawIndexedIndirectCommand& c : cmds) { ic = std::max(ic, c.indexCount); inst = std::max(inst, c.instanceCount); }
                std::string line;
                for (const DeviceData::PipeInfo::Bind& b : pi.binds) { char t[64]; snprintf(t, sizeof t, " bind%u(stride %u,%s)", b.binding, b.stride, b.rate ? "INSTANCE" : "vertex"); line += t; }
                line += " |";
                for (const DeviceData::PipeInfo::Attr& a : pi.attrs) { char t[96]; snprintf(t, sizeof t, " loc%u:b%u:%s@%u", a.loc, a.binding, FormatName(a.format), a.offset); line += t; }
                Logf("rt: LAYOUT pipe %p vs %016llx fs %016llx %s indexCount %u instances %u |%s",
                     (void*)st.pipe, (unsigned long long)pi.vsHash, (unsigned long long)pi.fsHash, pi.opaque ? "opaque" : "blended", ic, inst, line.c_str());
            }
        }
    }
    {   // step 17 diagnostic: raw per-instance data of instanced pipelines (trees / props) - needed to put them into the acceleration structure
        static std::mutex im;
        static std::unordered_map<uint64_t, int> perVs;
        uint32_t ib = 0, istride = 0;
        uint64_t vsh = 0;
        bool inst = false;
        {
            std::lock_guard<std::mutex> lk(d->regMtx);
            auto pit = d->pipes.find(st.pipe);
            if (pit != d->pipes.end()) {
                vsh = pit->second.vsHash;
                for (const DeviceData::PipeInfo::Bind& bd : pit->second.binds) if (bd.rate) { inst = true; ib = bd.binding; istride = bd.stride; }
            }
        }
        if (inst && istride >= 16 && istride <= 128 && ib < 8 && st.vb[ib] && !cmds.empty()) {
            std::lock_guard<std::mutex> lk(im);
            int& cnt = perVs[vsh];
            if (cnt < 5) {
                ++cnt;
                const VkDrawIndexedIndirectCommand& c0 = cmds[0];
                float cam[3] = {};
                {
                    DeviceData::UboRef ub{};
                    bool have = false;
                    std::lock_guard<std::mutex> ul(d->uboMtx);
                    auto it = d->mainUbo.find(cb);
                    if (it != d->mainUbo.end()) { ub = it->second; have = true; }
                    else if (!d->uboCands.empty()) { ub = DeviceData::UboRef{d->uboCands[0].buf, d->uboCands[0].off}; have = true; }
                    if (have) ReadBufHost(d, ub.buf, ub.off + 608, cam, sizeof cam);
                }
                Logf("rt: INSTANCE DATA vs %016llx stride %u | draw: indexCount %u instanceCount %u firstInstance %u vertexOffset %d | buffer %p off %llu | camera %.2f %.2f %.2f",
                     (unsigned long long)vsh, istride, c0.indexCount, c0.instanceCount, c0.firstInstance, (int)c0.vertexOffset, (void*)st.vb[ib],
                     (unsigned long long)st.vbOff[ib], cam[0], cam[1], cam[2]);
                for (uint32_t k = 0; k < 3 && k < std::max(1u, c0.instanceCount); ++k) {
                    uint8_t raw[128] = {};
                    if (!ReadBufHost(d, st.vb[ib], st.vbOff[ib] + (VkDeviceSize)(c0.firstInstance + k) * istride, raw, istride)) {
                        Logf("rt:   instance %u: instance buffer is not readable by the CPU", k);
                        break;
                    }
                    std::string line;
                    for (uint32_t w = 0; w + 4 <= istride; w += 4) {
                        uint32_t u; float f;
                        memcpy(&u, raw + w, 4); memcpy(&f, raw + w, 4);
                        char t[48];
                        snprintf(t, sizeof t, " @%u:%08x(%.3g)", w, u, std::isfinite(f) ? f : 0.0f);
                        line += t;
                    }
                    Logf("rt:   instance %u:%s", k, line.c_str());
                }
            }
        }
    }
    // vertex format of the pipeline: float3 position at location 0, per-vertex, opaque, not instanced
    bool fmtOk = false, isWater = false, isProp = false;
    uint32_t binding = 0, vstride = 0, voff = 0, propIb = 0, propIstride = 0;
    {
        std::lock_guard<std::mutex> lk(d->regMtx);
        auto pit = d->pipes.find(st.pipe);
        isWater = pit != d->pipes.end() && pit->second.vsHash == kWaterVs && pit->second.fsHash == kWaterFs;
        if (pit != d->pipes.end() && g_instGeo && g_dynGeo && pit->second.opaque && pit->second.instanced) {   // step 18: decoded instanced shaders only
            for (uint64_t kv : kInstVs) if (kv == pit->second.vsHash) isProp = true;
            if (isProp) for (const DeviceData::PipeInfo::Bind& bd : pit->second.binds) if (bd.rate) { propIb = bd.binding; propIstride = bd.stride; }
            if (propIb >= 8 || propIstride < 16) isProp = false;
        }
        if (pit != d->pipes.end() && (pit->second.opaque || (isWater && g_reflOn)) && (!pit->second.instanced || isProp)) {
            for (const DeviceData::PipeInfo::Attr& a : pit->second.attrs) {
                if (a.loc != 0 || a.format != (uint32_t)VK_FORMAT_R32G32B32_SFLOAT) continue;
                for (const DeviceData::PipeInfo::Bind& b : pit->second.binds)
                    if (b.binding == a.binding && b.rate == 0) { fmtOk = true; binding = a.binding; vstride = b.stride; voff = a.offset; }
            }
        }
    }
    DeviceData::BufInfo vbi{}, ibi{};
    VkDeviceAddress va = 0, ia = 0;
    const bool bufsOk = fmtOk && binding < 8 && st.vb[binding] && st.ib && st.ibType == VK_INDEX_TYPE_UINT16 && vstride >= 12 &&
                        BufInfoOf(d, st.vb[binding], vbi) && BufInfoOf(d, st.ib, ibi);
    if (bufsOk) {
        VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        ai.buffer = st.vb[binding];
        va = d->GetBufferDeviceAddress(d->device, &ai);
        ai.buffer = st.ib;
        ia = d->GetBufferDeviceAddress(d->device, &ai);
    }

    bool dynVb = false;   // step 16
    if (bufsOk && g_dynGeo && !isWater) {
        std::lock_guard<std::mutex> lk(d->regMtx);
        auto mit = d->mems.find(vbi.mem);
        dynVb = mit != d->mems.end() && mit->second.hostVisible && !(vbi.usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        static std::atomic<int> dl{0};
        if (dynVb && dl.fetch_add(1) < 6) {
            auto pit = d->pipes.find(st.pipe);
            Logf("rt: dynamic geometry: vertex buffer %p (usage 0x%x, host visible memory) drawn with vs %016llx - will be rebuilt every frame",
                 (void*)st.vb[binding], (unsigned)vbi.usage, pit != d->pipes.end() ? (unsigned long long)pit->second.vsHash : 0ull);
        }
    }
    std::lock_guard<std::mutex> lk(d->geoMtx);
    DeviceData::FrameGeo& fg = d->frameGeo[cb];
    fg.draws += indirect ? 1 : 1;
    if (unreadable) { fg.unreadable++; return; }
    if (!bufsOk || !va || !ia) { fg.skipPipe += cmds.size(); return; }
    for (const VkDrawIndexedIndirectCommand& c : cmds) {
        if (!c.indexCount || !c.instanceCount) continue;
        if (fg.geoms.size() >= 20000 || c.indexCount < 3 || c.vertexOffset < 0) { fg.skipOther++; continue; }
        const uint64_t vbase = st.vbOff[binding] + voff + (uint64_t)c.vertexOffset * vstride;
        const uint64_t ibase = st.ibOff + (uint64_t)c.firstIndex * 2;
        if (vbase + vstride > vbi.size || ibase + (uint64_t)c.indexCount * 2 > ibi.size) { fg.skipRange++; continue; }
        const uint64_t avail = (vbi.size - vbase) / vstride;
        VkAccelerationStructureGeometryKHR g{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        g.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        if (isWater) { static std::atomic<uint64_t> wn{0}; const uint64_t k = ++wn; if (k == 1 || k == 5000) Logf("rt: water geometry recognised by shader fingerprint (draw #%llu, %u indices, vertex stride %u)", (unsigned long long)k, c.indexCount, vstride); }
        g.flags = isWater ? 0 : VK_GEOMETRY_OPAQUE_BIT_KHR;   // water = the only non-opaque geometry: main rays cull it, the water ray looks only at it
        g.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        g.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        g.geometry.triangles.vertexData.deviceAddress = va + vbase;
        g.geometry.triangles.vertexStride = vstride;
        g.geometry.triangles.maxVertex = (uint32_t)(avail ? avail - 1 : 0);
        g.geometry.triangles.indexType = VK_INDEX_TYPE_UINT16;
        g.geometry.triangles.indexData.deviceAddress = ia + ibase;
        VkAccelerationStructureBuildRangeInfoKHR rg{};
        rg.primitiveCount = c.indexCount / 3;
        if (isProp) {   // step 18: one geometry per instance, same mesh, transform = position + uniform scale decoded from the instance buffer
            static std::atomic<uint64_t> pn{0}, pbad{0};
            if (c.instanceCount > kInstMaxPerDraw || c.indexCount < kInstMinIdx) { fg.skipOther++; continue; }
            for (uint32_t k = 0; k < c.instanceCount; ++k) {
                if (fg.propGeoms >= kPropGeomMax || fg.propTris + rg.primitiveCount > kPropTrisMax || fg.dynGeoms.size() >= kXfCap) break;
                float iv[4] = {};
                if (!ReadBufHost(d, st.vb[propIb], st.vbOff[propIb] + (VkDeviceSize)(c.firstInstance + k) * propIstride, iv, sizeof iv)) { ++pbad; break; }
                float sc = iv[3] * g_instScale;
                if (!(sc > 0.0005f && sc < 100.0f)) sc = g_instScale;
                if (!(fabsf(iv[0]) < 1e6f && fabsf(iv[1]) < 1e6f && fabsf(iv[2]) < 1e6f)) { ++pbad; continue; }
                VkTransformMatrixKHR xf{};
                xf.matrix[0][0] = xf.matrix[1][1] = xf.matrix[2][2] = sc;
                xf.matrix[0][3] = iv[0]; xf.matrix[1][3] = iv[1]; xf.matrix[2][3] = iv[2];
                fg.dynGeoms.push_back(g);
                fg.dynRanges.push_back(rg);
                fg.dynKeys.push_back(GeoKey{st.vb[binding], st.ib, vbase, ibase, c.indexCount, vstride});
                fg.dynMems.emplace_back(vbi.mem, ibi.mem);
                fg.dynXf.push_back(xf);
                fg.dynProp.push_back(1);
                fg.propTris += rg.primitiveCount;
                fg.propGeoms++;
                const uint64_t n = ++pn;
                if (n <= 3 || n == 20000) Logf("rt: prop instance #%llu: pos %.2f %.2f %.2f scale %.3f | indexCount %u (instance buffer %p)", (unsigned long long)n, iv[0], iv[1], iv[2], sc, c.indexCount, (void*)st.vb[propIb]);
            }
            continue;
        }
        if (dynVb) {   // step 16: CPU-written vertex data changes every frame -> not cached, own per-frame BLAS
            if (fg.dynGeoms.size() - fg.propGeoms < 1024) {
                fg.dynXf.push_back(VkTransformMatrixKHR{{{1,0,0,0},{0,1,0,0},{0,0,1,0}}});
                fg.dynProp.push_back(0);
                fg.dynGeoms.push_back(g);
                fg.dynRanges.push_back(rg);
                fg.dynKeys.push_back(GeoKey{st.vb[binding], st.ib, vbase, ibase, c.indexCount, vstride});
                fg.dynMems.emplace_back(vbi.mem, ibi.mem);
                fg.dynTris += rg.primitiveCount;
                d->dynClassified++;
            }
            continue;
        }
        fg.geoms.push_back(g);
        fg.ranges.push_back(rg);
        fg.keys.push_back(GeoKey{st.vb[binding], st.ib, vbase, ibase, c.indexCount, vstride});
        fg.mems.emplace_back(vbi.mem, ibi.mem);
        fg.tris += rg.primitiveCount;
    }
}

// ---------------------------------------------------------------- camera
static bool FiniteMat(const float* m, int n) {
    for (int i = 0; i < n; ++i) if (!(m[i] > -1e9f && m[i] < 1e9f)) return false;
    return true;
}

static bool Invert4(const float* in, float* out) {  // column-major in / out, double precision Gauss-Jordan
    double a[4][8];
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            a[r][c] = in[c * 4 + r];
            a[r][4 + c] = (r == c) ? 1.0 : 0.0;
        }
    for (int c = 0; c < 4; ++c) {
        int piv = c;
        for (int r = c + 1; r < 4; ++r) if (fabs(a[r][c]) > fabs(a[piv][c])) piv = r;
        if (fabs(a[piv][c]) < 1e-30) return false;
        if (piv != c) for (int k = 0; k < 8; ++k) std::swap(a[piv][k], a[c][k]);
        const double inv = 1.0 / a[c][c];
        for (int k = 0; k < 8; ++k) a[c][k] *= inv;
        for (int r = 0; r < 4; ++r) {
            if (r == c) continue;
            const double f = a[r][c];
            if (f == 0.0) continue;
            for (int k = 0; k < 8; ++k) a[r][k] -= f * a[c][k];
        }
    }
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) out[c * 4 + r] = (float)a[r][4 + c];
    return true;
}

static void DumpUnitVectors(DeviceData* d, const DeviceData::UboRef& u) {
    std::vector<float> f(1584 / 4);
    if (!ReadBufHost(d, u.buf, u.off, f.data(), 1584)) { Logf("rt: UBO dump: not readable"); return; }
    Logf("rt: frame UBO scan for unit-length float3 rows (light direction candidates):");
    int n = 0;
    for (size_t row = 0; row + 3 < f.size() && n < 40; row += 4) {
        const float* v = &f[row];
        const float len = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (len > 0.98f && len < 1.02f) {
            Logf("    offset %4zu: %.4f %.4f %.4f | w=%.4f", row * 4, v[0], v[1], v[2], v[3]);
            n++;
        }
    }
}

static void LoadCfg(bool logIt) {  // SkyRT.cfg next to the DLL, re-read while the game runs
    char path[MAX_PATH];
    ModuleDir(path, sizeof path);
    strncat(path, "\\SkyRT.cfg", sizeof path - strlen(path) - 1);
    FILE* f = fopen(path, "rb");
    if (!f) {
        f = fopen(path, "wb");
        if (f) {
            fputs("# SkyRT live settings. Edit and save while the game is running - applied within about a second.\r\n"
                  "# In game: Ctrl+Home = ray tracing on/off (original picture vs ours), Ctrl+End = next view, Ctrl+PageDown = probe water/ice draws (log only).\r\n"
                  "enabled=1     # 1 = ray tracing on, 0 = the original game picture\r\n"
                  "view=0        # 0 full, 1 shadows only, 2 AO only, 3 bounce light only, 4 DEBUG AO map, 5 DEBUG bounce map, 6 DEBUG grid, 7 DEBUG normals\r\n"
                  "shadows=1     # sun shadows by rays (used by view 0)\r\n"
                  "ao=1          # ambient occlusion by rays (used by view 0)\r\n"
                  "gi=1          # bounce light / color bleeding by rays (used by view 0)\r\n"
                  "strength=0.5  # shadow darkness 0..1\r\n"
                  "sunsize=2.5   # degrees, soft shadow penumbra (0 = hard shadows)\r\n"
                  "shrays=2      # shadow rays per pixel 1..8\r\n"
                  "aostrength=0.7  # 0..1\r\n"
                  "aoradius=1.5  # world units, how far an occluder can be\r\n"
                  "aorays=4      # rays per pixel 1..8 for AO and bounce light (more = less noise, slower)\r\n"
                  "gistrength=0.35 # 0..2\r\n"
                  "girange=30    # world units, how far bounce rays look\r\n"
                  "watervs=17a63a0f3447fc7c  # shader fingerprints of the water pipeline (hex)\r\n"
                  "waterfs=22efde0f73a797a4\r\n"
                  "waterfx=0     # EXPERIMENTAL ray traced water (do not enable: the real water pipeline is not found yet): reflections of the world and sun glints\r\n"
                  "waterrefl=0.5 # 0..1 how strongly the world is mirrored in water\r\n"
                  "glint=1.0     # 0..4 sun glitter on the water\r\n"
                  "taa=1         # 1 = temporal accumulation of shadows / AO / GI (less noise)\r\n"
                  "taan=12       # max accumulated frames (lower = less ghosting, more noise)\r\n"
                  "light=1       # 1 = light from fire, candles and lamps (bright pixels become light sources that cast shadows)\r\n"
                  "lightstrength=1.0 # 0..10 how strong that light is\r\n"
                  "lightrange=12 # world units, how far the light of one source reaches\r\n"
                  "lightthr=2.0  # brightness above which a pixel counts as a light source (the log prints the brightest pixel it sees)\r\n"
                  "lightrays=2   # shadow rays per pixel for those lights 1..4\r\n"
                  "lightdebug=0  # 1 = paint the pixels that are treated as light sources magenta (to tune lightthr)\r\n"
                  "instgeo=1     # 1 = instanced props in the acceleration structure\r\n"
                  "dyngeo=1      # 1 = characters / animated meshes get their own acceleration structure rebuilt every frame (smooth shadows of moving objects)\r\n"
                  "aniso=16      # anisotropic filtering forced on all textures: 0 = leave the game alone, 2/4/8/16 (applies on next game start)\r\n"
                  "lodbias=0     # -2..1 texture sharpness shift, negative = sharper (try -0.5), applies on next game start\r\n"
                  "autosun=1     # 1 = sun direction from the game, 0 = use az / el below\r\n"
                  "az=30         # sun azimuth in degrees (used when autosun=0)\r\n"
                  "el=50         # sun elevation in degrees (used when autosun=0)\r\n", f);
            fclose(f);
        }
        return;
    }
    char buf[4096] = {};
    const size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = 0;
    for (char* line = strtok(buf, "\r\n"); line; line = strtok(nullptr, "\r\n")) {
        char* c = strchr(line, '#');
        if (c) *c = 0;
        char* eq = strchr(line, '=');
        if (!eq) continue;
        *eq = 0;
        const char* k = line;
        const double v = atof(eq + 1);
        const char* sv = eq + 1;
        while (*k == ' ' || *k == '\t') ++k;
        char key[32] = {};
        strncpy(key, k, sizeof key - 1);
        for (int i = (int)strlen(key) - 1; i >= 0 && (key[i] == ' ' || key[i] == '\t'); --i) key[i] = 0;
        if (!strcmp(key, "view")) g_view = (uint32_t)v % 8u;
        else if (!strcmp(key, "enabled")) g_enabled = v != 0.0;
        else if (!strcmp(key, "autosun")) g_autoSun = v != 0.0;
        else if (!strcmp(key, "az")) g_sunAz = v;
        else if (!strcmp(key, "el")) g_sunEl = std::max(1.0, std::min(89.0, v));
        else if (!strcmp(key, "strength")) g_strength = (float)std::max(0.0, std::min(1.0, v));
        else if (!strcmp(key, "shadows")) g_shadows = v != 0.0;
        else if (!strcmp(key, "ao")) g_aoOn = v != 0.0;
        else if (!strcmp(key, "aostrength")) g_aoStrength = (float)std::max(0.0, std::min(1.0, v));
        else if (!strcmp(key, "aoradius")) g_aoRadius = (float)std::max(0.05, std::min(20.0, v));
        else if (!strcmp(key, "aorays")) g_aoRays = (int)std::max(1.0, std::min(8.0, v));
        else if (!strcmp(key, "gi")) g_giOn = v != 0.0;
        else if (!strcmp(key, "gistrength")) g_giStrength = (float)std::max(0.0, std::min(2.0, v));
        else if (!strcmp(key, "girange")) g_giRange = (float)std::max(1.0, std::min(200.0, v));
        else if (!strcmp(key, "sunsize")) g_sunSize = (float)std::max(0.0, std::min(10.0, v));
        else if (!strcmp(key, "shrays")) g_shRays = (int)std::max(1.0, std::min(8.0, v));
        else if (!strcmp(key, "watervs")) kWaterVs = strtoull(sv, nullptr, 16);
        else if (!strcmp(key, "waterfs")) kWaterFs = strtoull(sv, nullptr, 16);
        else if (!strcmp(key, "waterfx")) g_reflOn = v != 0.0;
        else if (!strcmp(key, "waterrefl")) g_reflStrength = (float)std::max(0.0, std::min(1.0, v));
        else if (!strcmp(key, "glint")) g_glint = (float)std::max(0.0, std::min(4.0, v));
        else if (!strcmp(key, "dumpshaders")) g_dumpShaders = v != 0.0;
        else if (!strcmp(key, "dyngeo")) g_dynGeo = v != 0.0;
        else if (!strcmp(key, "taa")) g_taa = v != 0.0;
        else if (!strcmp(key, "taan")) g_taaN = (float)std::max(1.0, std::min(64.0, v));
        else if (!strcmp(key, "light")) g_lightOn = v != 0.0;
        else if (!strcmp(key, "lightstrength")) g_lightStrength = (float)std::max(0.0, std::min(10.0, v));
        else if (!strcmp(key, "lightrange")) g_lightRange = (float)std::max(1.0, std::min(60.0, v));
        else if (!strcmp(key, "lightthr")) g_lightThr = (float)std::max(0.3, std::min(50.0, v));
        else if (!strcmp(key, "lightrays")) g_lightRays = (int)std::max(1.0, std::min(4.0, v));
        else if (!strcmp(key, "lightdebug")) g_lightDebug = v != 0.0;
        else if (!strcmp(key, "instgeo")) g_instGeo = v != 0.0;
        else if (!strcmp(key, "instscale")) g_instScale = (float)v;
        else if (!strcmp(key, "aniso")) g_aniso = (int)std::max(0.0, std::min(16.0, v));
        else if (!strcmp(key, "lodbias")) g_lodBias = (float)std::max(-2.0, std::min(1.0, v));
    }
    if (logIt)
        Logf("rt: settings (SkyRT.cfg): enabled=%d view=%u (%s) | shadows=%d (strength %.2f, sun %.1f deg, %d rays) ao=%d (strength %.2f radius %.2f) gi=%d (strength %.2f range %.0f) hemisphere rays %d | autosun=%d az=%.1f el=%.1f",
             g_enabled ? 1 : 0, g_view, kViewNames[g_view % 8], g_shadows ? 1 : 0, (double)g_strength, (double)g_sunSize, g_shRays, g_aoOn ? 1 : 0,
             (double)g_aoStrength, (double)g_aoRadius, g_giOn ? 1 : 0, (double)g_giStrength, (double)g_giRange, g_aoRays, g_autoSun ? 1 : 0, g_sunAz, g_sunEl);
    if (logIt)
        Logf("rt: emissive light: light=%d strength %.2f range %.1f threshold %.2f rays %d debug %d", g_lightOn ? 1 : 0, (double)g_lightStrength, (double)g_lightRange, (double)g_lightThr, g_lightRays, g_lightDebug ? 1 : 0);
}

static void PollKeys(uint64_t frame) {  // Ctrl+Home = on/off, Ctrl+End = next view; SkyRT.cfg is re-read when it changes
    if (frame % 60 == 1) {
        static FILETIME lastWrite = {};
        char path[MAX_PATH];
        ModuleDir(path, sizeof path);
        strncat(path, "\\SkyRT.cfg", sizeof path - strlen(path) - 1);
        WIN32_FILE_ATTRIBUTE_DATA fa{};
        if (!GetFileAttributesExA(path, GetFileExInfoStandard, &fa)) LoadCfg(true);
        else if (fa.ftLastWriteTime.dwLowDateTime != lastWrite.dwLowDateTime || fa.ftLastWriteTime.dwHighDateTime != lastWrite.dwHighDateTime) {
            lastWrite = fa.ftLastWriteTime;
            LoadCfg(true);
        }
    }
    static bool alive = false;
    const bool ctrl = (GetAsyncKeyState(VK_CONTROL) & 0x8000) != 0;
    static bool homeWas = false, endWas = false;
    const bool home = ctrl && (GetAsyncKeyState(VK_HOME) & 0x8000) != 0;
    const bool end = ctrl && (GetAsyncKeyState(VK_END) & 0x8000) != 0;
    if (ctrl && !alive) { alive = true; Logf("rt: keyboard polling works (Ctrl seen). Ctrl+Home = ray tracing on/off, Ctrl+End = next view"); }
    if (home && !homeWas) {
        g_enabled = !g_enabled;
        Logf("rt: KEY Ctrl+Home -> ray tracing %s", g_enabled ? "ON (our render)" : "OFF (original game picture)");
    }
    if (end && !endWas) {
        g_view = (g_view + 1) % 8u;
        if (!g_enabled) g_enabled = true;
        Logf("rt: KEY Ctrl+End -> view %u: %s", g_view, kViewNames[g_view]);
    }
    static bool pdWas = false;
    const bool pd = ctrl && (GetAsyncKeyState(VK_NEXT) & 0x8000) != 0;
    static bool puWas = false;
    const bool pu = ctrl && (GetAsyncKeyState(VK_PRIOR) & 0x8000) != 0;
    if (pd && !pdWas) ProbeKey(false);
    if (pu && !puWas) ProbeKey(true);
    puWas = pu;
    static bool delWas = false;
    const bool del = ctrl && (GetAsyncKeyState(VK_DELETE) & 0x8000) != 0;
    if (del && !delWas) PassProbeKey();
    delWas = del;
    if (pu && g_ppState.load()) { g_ppState = 0; g_ppHide = -1; Logf("rt: pass probe stopped"); }
    {   // step 12: manual stepping while a probe is frozen: Ctrl+Right = next, Ctrl+Left = previous
        static bool rWas = false, lWas = false;
        const bool r = ctrl && (GetAsyncKeyState(VK_RIGHT) & 0x8000) != 0;
        const bool l = ctrl && (GetAsyncKeyState(VK_LEFT) & 0x8000) != 0;
        const int dir = (r && !rWas) ? 1 : ((l && !lWas) ? -1 : 0);
        rWas = r; lWas = l;
        if (dir) {
            if (g_ppState.load() == 3) {
                size_t n; { std::lock_guard<std::mutex> lk(g_ppMtx); n = g_ppList.size(); }
                if (n) { int h = g_ppHide.load() + dir; if (h < 0) h = (int)n - 1; if ((size_t)h >= n) h = 0; g_ppHide = h; PassProbeLog(h); }
            }
            if (g_probeState.load() == 3) {
                size_t n; { std::lock_guard<std::mutex> lk(g_probeMtx); n = g_probeList.size(); }
                if (n) { int h = g_hide.load() + dir; if (h < 0) h = (int)n - 1; if ((size_t)h >= n) h = 0; g_hide = h; ProbeLogHide(h); }
            }
        }
    }
    {   // step 12: Ctrl+Insert = hide a whole GROUP of blended pipelines at once (the water may be several similar pipelines)
        static bool insWas = false;
        const bool ins = ctrl && (GetAsyncKeyState(VK_INSERT) & 0x8000) != 0;
        if (ins && !insWas) {
            g_probeOn = true;
            const int g = (g_hideGroup.load() + 1) % 5;
            g_hideGroup = g;
            g_hide = -1;
            Logf("rt: KEY Ctrl+Insert -> GROUP HIDE %d: %s", g, kGroupNames[g]);
        }
        insWas = ins;
    }
    ProbeTick(frame);
    PassProbeTick(frame);
    pdWas = pd;
    homeWas = home;
    endWas = end;
}

// called after vkCmdBeginRenderPass: remembers command buffers that entered the main scene pass
static void MainPassBegin(DeviceData* d, VkCommandBuffer cb, const VkRenderPassBeginInfo* pBegin) {
    if (g_logOnly || g_noPaint || !d->rtEnabled || !pBegin) return;
    VkImage color = VK_NULL_HANDLE, depth = VK_NULL_HANDLE;
    uint32_t w = 0, h = 0;
    bool looksMain = false, usable = false;
    {
        std::lock_guard<std::mutex> lk(d->regMtx);
        auto rit = d->rps.find(pBegin->renderPass);
        if (rit == d->rps.end() || rit->second.size() != 2) return;
        const std::vector<DeviceData::AttInfo>& a = rit->second;
        if (a[0].format != (uint32_t)VK_FORMAT_R16G16B16A16_SFLOAT || a[0].loadOp != (uint32_t)VK_ATTACHMENT_LOAD_OP_CLEAR ||
            a[1].format != (uint32_t)VK_FORMAT_D32_SFLOAT || a[1].loadOp != (uint32_t)VK_ATTACHMENT_LOAD_OP_CLEAR)
            return;
        auto fit = d->fbs.find(pBegin->framebuffer);
        if (fit == d->fbs.end() || fit->second.views.size() != 2) return;
        auto v0 = d->views.find(fit->second.views[0]);
        auto v1 = d->views.find(fit->second.views[1]);
        if (v0 == d->views.end() || v1 == d->views.end()) return;
        auto i0 = d->imgs.find(v0->second.image);
        auto i1 = d->imgs.find(v1->second.image);
        if (i0 == d->imgs.end() || i1 == d->imgs.end()) return;
        looksMain = i0->second.w >= 800 && i0->second.w == i1->second.w && i0->second.h == i1->second.h;
        if (!looksMain) return;
        usable = (i0->second.patchedUsage & VK_IMAGE_USAGE_STORAGE_BIT) && (i1->second.patchedUsage & VK_IMAGE_USAGE_SAMPLED_BIT);
        color = v0->second.image;
        depth = v1->second.image;
        w = i0->second.w;
        h = i0->second.h;
    }
    if (!usable) {
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 3) Logf("paint: main pass found (%ux%u) but its images lack STORAGE/SAMPLED usage - skipped", w, h);
        return;
    }
    std::lock_guard<std::mutex> lk(d->paintMtx);
    if (d->mainCb.emplace(cb, DeviceData::MainCb{color, depth, w, h}).second) d->mainCount++;
    else d->mainCb[cb] = DeviceData::MainCb{color, depth, w, h};
}

// ---------------------------------------------------------------- step 16: per-frame dynamic BLAS (characters) + a two-instance TLAS (static world + dynamic)
static const VkDeviceSize kDynBlasCap = 40ull << 20, kDynScratchCap = 24ull << 20;
static bool EnsureDynSlots(DeviceData* d) {
    if (d->dynTried) return d->dynOk;
    d->dynTried = true;
    VkAccelerationStructureGeometryKHR tg{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    tg.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tg.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    VkAccelerationStructureBuildGeometryInfoKHR tb{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    tb.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    tb.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    tb.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    tb.geometryCount = 1;
    tb.pGeometries = &tg;
    const uint32_t two = 2;
    VkAccelerationStructureBuildSizesInfoKHR tsz{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    d->GetAccelerationStructureBuildSizesKHR(d->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tb, &two, &tsz);
    const VkDeviceSize tlasCap = (tsz.accelerationStructureSize + 255) & ~255ull;
    for (int i = 0; i < 8; ++i) {
        DeviceData::DynSlot& sl = d->dyn[i];
        VkDeviceAddress dummy = 0;
        const char* failed = nullptr;
        if (!MakeBufEx(d, kDynBlasCap, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, sl.blasBuf, sl.blasMem, dummy)) failed = "dyn BLAS storage";
        else if (!MakeBufEx(d, tlasCap, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, sl.tlasBuf, sl.tlasMem, dummy)) failed = "dyn TLAS storage";
        else if (!MakeBufEx(d, kDynScratchCap + d->asScratchAlign, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, sl.scratchBuf, sl.scratchMem, sl.scratchAddr)) failed = "dyn scratch";
        else if (!MakeBufEx(d, 4096, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, sl.instBuf, sl.instMem, sl.instAddr)) failed = "dyn instance buffer";
        else if (!MakeBufEx(d, (VkDeviceSize)kXfCap * 48, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, sl.xfBuf, sl.xfMem, sl.xfAddr)) failed = "dyn transform buffer";
        if (!failed) {
            VkAccelerationStructureCreateInfoKHR aci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
            aci.buffer = sl.blasBuf; aci.size = kDynBlasCap; aci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            if (d->CreateAccelerationStructureKHR(d->device, &aci, nullptr, &sl.blas) != VK_SUCCESS) failed = "dyn BLAS object";
            else {
                aci.buffer = sl.tlasBuf; aci.size = tlasCap; aci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
                if (d->CreateAccelerationStructureKHR(d->device, &aci, nullptr, &sl.tlas) != VK_SUCCESS) failed = "dyn TLAS object";
            }
        }
        if (!failed) {
            VkAccelerationStructureDeviceAddressInfoKHR ai{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
            ai.accelerationStructure = sl.blas;
            sl.blasAddr = d->GetAccelerationStructureDeviceAddressKHR(d->device, &ai);
            void* mp = nullptr;
            if (!sl.blasAddr || d->MapMemory(d->device, sl.instMem, 0, VK_WHOLE_SIZE, 0, &mp) != VK_SUCCESS || !mp) failed = "dyn map/address";
            else sl.instPtr = mp;
            void* xp = nullptr;
            if (!failed && (d->MapMemory(d->device, sl.xfMem, 0, VK_WHOLE_SIZE, 0, &xp) != VK_SUCCESS || !xp)) failed = "dyn transform map";
            else sl.xfPtr = xp;
        }
        if (failed) { Logf("!!! rt: dynamic slot %d: failed to create %s - dynamic geometry disabled", i, failed); return false; }
        sl.ok = true;
    }
    d->dynOk = true;
    Logf("rt: 8 dynamic AS slots ready (BLAS %llu MB, scratch %llu MB each)", (unsigned long long)(kDynBlasCap >> 20), (unsigned long long)(kDynScratchCap >> 20));
    return true;
}

static void InjectPaint(DeviceData* d, VkCommandBuffer cb, const DeviceData::MainCb& m) {
    g_stage = "InjectRT";
    DeviceData::FrameGeo fg;
    {
        std::lock_guard<std::mutex> lk(d->geoMtx);
        auto it = d->frameGeo.find(cb);
        if (it != d->frameGeo.end()) { fg = std::move(it->second); d->frameGeo.erase(it); }
    }
    if (!fg.dynGeoms.empty() && (!g_dynGeo || fg.dynTris > kDynTrisMax)) {   // too much to be characters: treat as ordinary static geometry (cached)
        static std::atomic<int> fl{0};
        if (g_dynGeo && fl.fetch_add(1) < 5) Logf("rt: %zu 'dynamic' geometries / %llu tris is too much for characters - treated as static", fg.dynGeoms.size(), (unsigned long long)fg.dynTris);
        for (size_t i = 0; i < fg.dynGeoms.size(); ++i) {
            if (i < fg.dynProp.size() && fg.dynProp[i]) continue;   // props have no meaning without their transform
            fg.geoms.push_back(fg.dynGeoms[i]); fg.ranges.push_back(fg.dynRanges[i]);
            fg.keys.push_back(fg.dynKeys[i]); fg.mems.push_back(fg.dynMems[i]);
            fg.tris += fg.dynRanges[i].primitiveCount;
        }
        fg.dynGeoms.clear(); fg.dynRanges.clear(); fg.dynKeys.clear(); fg.dynMems.clear(); fg.dynXf.clear(); fg.dynProp.clear(); fg.dynTris = 0; fg.propTris = 0; fg.propGeoms = 0;
    }
    DeviceData::UboRef ubo{};
    bool haveUbo = false, haveUboBound = false;
    {
        std::lock_guard<std::mutex> lk(d->uboMtx);
        auto it = d->mainUbo.find(cb);
        if (it != d->mainUbo.end()) { ubo = it->second; haveUbo = true; haveUboBound = true; }
        else if (!d->uboCands.empty()) { ubo = DeviceData::UboRef{d->uboCands[0].buf, d->uboCands[0].off}; haveUbo = true; }
    }
    static std::atomic<int> skipLogs{0};
    auto skip = [&](const char* why) {
        d->rtSkipped++;
        if (skipLogs.fetch_add(1) < 12)
            Logf("rt: frame skipped: %s (geoms %zu, tris %llu, draws %llu)", why, fg.geoms.size(), (unsigned long long)fg.tris,
                 (unsigned long long)fg.draws);
    };
    std::vector<VkAccelerationStructureGeometryKHR> geoms;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
    uint64_t cacheTrisNow = 0;
    bool rebuild = false;
    int curSlot = -1;
    {
        std::lock_guard<std::mutex> lk(d->cacheMtx);
        const uint64_t fr = d->frames.load();
        for (size_t i = 0; i < fg.keys.size(); ++i) {
            auto it = d->geoCache.find(fg.keys[i]);
            if (it == d->geoCache.end()) {
                d->geoCache.emplace(fg.keys[i], GeoEnt{fg.geoms[i], fg.ranges[i], fr, fg.mems[i].first, fg.mems[i].second});
                d->buildMems.insert(fg.mems[i].first);
                d->buildMems.insert(fg.mems[i].second);
                d->cachedBufs.insert(fg.keys[i].vb);
                d->cachedBufs.insert(fg.keys[i].ib);
                d->cacheTris += fg.ranges[i].primitiveCount;
                d->cacheDirty = true;
            } else {
                it->second.lastSeen = fr;
            }
        }
        if (fr - d->lastEvict >= 300) {  // forget what has not been drawn for a minute
            d->lastEvict = fr;
            for (auto it = d->geoCache.begin(); it != d->geoCache.end();) {
                if (fr - it->second.lastSeen > 12000) {
                    d->cacheTris -= std::min<uint64_t>(d->cacheTris, it->second.r.primitiveCount);
                    it = d->geoCache.erase(it);
                    d->rtEvicted++;
                    d->cacheDirty = true;
                } else ++it;
            }
        }
        if (d->cacheTris > kMaxCacheTris || d->geoCache.size() > kMaxCacheGeoms) {  // drop the least recently seen
            std::vector<std::pair<uint64_t, GeoKey>> order;
            order.reserve(d->geoCache.size());
            for (auto& kv : d->geoCache) order.emplace_back(kv.second.lastSeen, kv.first);
            std::sort(order.begin(), order.end(), [](const std::pair<uint64_t, GeoKey>& a, const std::pair<uint64_t, GeoKey>& b) { return a.first < b.first; });
            for (auto& o : order) {
                if (d->cacheTris <= kMaxCacheTris * 3 / 4 && d->geoCache.size() <= kMaxCacheGeoms * 3 / 4) break;
                auto it = d->geoCache.find(o.second);
                d->cacheTris -= std::min<uint64_t>(d->cacheTris, it->second.r.primitiveCount);
                d->geoCache.erase(it);
                d->rtEvicted++;
            }
            d->cacheForce = true;
        }
        cacheTrisNow = d->cacheTris;
        curSlot = d->rtCur;
        if (d->geoCache.empty()) { d->cacheDirty = false; d->cacheForce = false; }
        else if (d->cacheForce || d->rtCur < 0 || (d->cacheDirty && fr - d->lastRebuildFrame >= RebuildGap(d->cacheTris, d->lastRebuildTris))) {
            rebuild = true;
            d->cacheDirty = false;
            d->cacheForce = false;
            d->lastRebuildFrame = fr;
            d->lastRebuildTris = d->cacheTris;
            d->lastRebuildFrameA = fr;
            geoms.reserve(d->geoCache.size());
            ranges.reserve(d->geoCache.size());
            for (auto& kv : d->geoCache) { geoms.push_back(kv.second.g); ranges.push_back(kv.second.r); }
        }
    }
    if (!g_enabled) {  // original picture: keep the cache up to date but do no GPU work
        if (rebuild) { std::lock_guard<std::mutex> lk(d->cacheMtx); d->cacheForce = true; }
        return;
    }
    if (!rebuild && curSlot < 0) { skip("no usable geometry recorded in the main pass"); return; }
    if (!haveUbo) { skip("frame UBO not found"); d->rtNoCam++; return; }

    // camera from the game's own UBO (viewProj @224, camera position @608)
    float vp[16] = {}, cam[3] = {};
    if (!ReadBufHost(d, ubo.buf, ubo.off + 224, vp, sizeof vp) || !ReadBufHost(d, ubo.buf, ubo.off + 608, cam, sizeof cam) ||
        !FiniteMat(vp, 16) || !FiniteMat(cam, 3)) { skip("frame UBO is not readable / not finite"); d->rtNoCam++; return; }
    float inv[16];
    if (!Invert4(vp, inv) || !FiniteMat(inv, 16)) { skip("viewProj is not invertible"); d->rtNoCam++; return; }

    VkImageView cv = VK_NULL_HANDLE, dv = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    uint32_t ringIdx = 0;
    DeviceData::RtSlot* slot = nullptr;
    {
        std::lock_guard<std::mutex> lk(d->paintMtx);
        if (!EnsurePaintResources(d) || !EnsureRtSlots(d)) return;
        cv = GetOwnView(d, m.color, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);
        dv = GetOwnView(d, m.depth, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT);
        if (!cv || !dv) return;
        ringIdx = d->paintRing.fetch_add(1) % 16;
        set = d->paintSets[ringIdx];
    }

    // ---- (re)build BLAS + TLAS only when the set of geometry changed, otherwise reuse the newest complete slot
    VkAccelerationStructureBuildSizesInfoKHR sz{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    int pickIdx = -1;
    if (rebuild) {  // never overwrite an AS that a frame still in flight may be tracing against
        std::lock_guard<std::mutex> lk(d->cacheMtx);
        const uint64_t fr = d->frames.load();
        uint64_t best = ~0ull;
        for (int i = 0; i < 8; ++i) {
            if (i == d->rtCur) continue;
            const uint64_t lu = d->rtSlots[i].lastUse;
            if (lu && fr - lu < 8) continue;
            if (lu < best) { best = lu; pickIdx = i; }
        }
        if (pickIdx < 0) d->cacheForce = true;  // try again next frame
    }
    if (rebuild && pickIdx < 0) {
        rebuild = false;
        static std::atomic<int> busyLogs{0};
        if (busyLogs.fetch_add(1) < 10) Logf("rt: all AS slots are still in use by earlier frames - keeping the old AS this frame");
        if (curSlot < 0) { skip("no AS slot is free"); return; }
    }
    if (rebuild) {
        const int idx = pickIdx;
        slot = &d->rtSlots[idx];
        VkAccelerationStructureBuildGeometryInfoKHR bi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        bi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        bi.geometryCount = (uint32_t)geoms.size();
        bi.pGeometries = geoms.data();
        std::vector<uint32_t> maxPrim(geoms.size());
        for (size_t i = 0; i < geoms.size(); ++i) maxPrim[i] = ranges[i].primitiveCount;
        d->GetAccelerationStructureBuildSizesKHR(d->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, maxPrim.data(), &sz);
        if (sz.accelerationStructureSize > slot->blasCap || sz.buildScratchSize > slot->scratchCap) {
            skip("BLAS / scratch bigger than the preallocated buffers");
            std::lock_guard<std::mutex> lk(d->cacheMtx);
            d->geoCache.clear();
            d->cacheTris = 0;
            d->cacheDirty = false;
            return;
        }

        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                           VK_ACCESS_SHADER_READ_BIT;
        d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1,
                              &mb, 0, nullptr, 0, nullptr);

        const VkDeviceAddress scratch = (slot->scratchAddr + d->asScratchAlign - 1) & ~(d->asScratchAlign - 1);
        bi.dstAccelerationStructure = slot->blas;
        bi.scratchData.deviceAddress = scratch;
        const VkAccelerationStructureBuildRangeInfoKHR* pRanges = ranges.data();
        d->CmdBuildAccelerationStructuresKHR(cb, 1, &bi, &pRanges);

        VkMemoryBarrier asb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        asb.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        asb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                              VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &asb, 0, nullptr, 0, nullptr);

        VkAccelerationStructureGeometryKHR tg{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        tg.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        tg.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        tg.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        tg.geometry.instances.arrayOfPointers = VK_FALSE;
        tg.geometry.instances.data.deviceAddress = slot->instAddr;
        VkAccelerationStructureBuildGeometryInfoKHR tb{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        tb.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        tb.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        tb.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        tb.geometryCount = 1;
        tb.pGeometries = &tg;
        tb.dstAccelerationStructure = slot->tlas;
        tb.scratchData.deviceAddress = scratch;
        VkAccelerationStructureBuildRangeInfoKHR trange{};
        trange.primitiveCount = 1;
        const VkAccelerationStructureBuildRangeInfoKHR* pTr = &trange;
        d->CmdBuildAccelerationStructuresKHR(cb, 1, &tb, &pTr);
        {
            std::lock_guard<std::mutex> lk(d->cacheMtx);
            d->rtCur = idx;
        }
        const uint64_t nb = ++d->rtRebuilds;
        if (nb <= 30 || nb % 200 == 0)
            Logf("rt: BLAS rebuild #%llu: %zu geometries, %llu triangles (cache), BLAS %.2f MB, scratch %.2f MB | this frame drew %zu geometries / %llu tris",
                 (unsigned long long)nb, geoms.size(), (unsigned long long)cacheTrisNow, sz.accelerationStructureSize / 1048576.0,
                 sz.buildScratchSize / 1048576.0, fg.geoms.size(), (unsigned long long)fg.tris);
    } else {
        slot = &d->rtSlots[curSlot];
    }
    {
        std::lock_guard<std::mutex> lk(d->cacheMtx);
        slot->lastUse = d->frames.load();
    }
    VkAccelerationStructureKHR traceTlas = slot->tlas;
    if (g_dynGeo && !fg.dynGeoms.empty() && slot->blasAddr) {   // step 16: characters = own BLAS every frame, TLAS = static world + dynamic
        d->dynFrames++;
        d->dynGeomsSum += fg.dynGeoms.size();
        d->dynTrisSum += fg.dynTris;
        DeviceData::DynSlot* ds = nullptr;
        const char* why = nullptr;
        if (fg.dynTris > kDynTrisMax) why = "too many triangles";
        else if (![&] { std::lock_guard<std::mutex> lk0(d->paintMtx); return EnsureDynSlots(d); }()) why = "slots unavailable";
        else {
            std::lock_guard<std::mutex> lk(d->cacheMtx);
            const uint64_t fr = d->frames.load();
            uint64_t best = ~0ull;
            for (int i = 0; i < 8; ++i) {
                const uint64_t lu = d->dyn[i].lastUse;
                if (lu && fr - lu < 6) continue;
                if (lu < best) { best = lu; ds = &d->dyn[i]; }
            }
            if (!ds) why = "all dynamic slots busy";
        }
        if (!why) {
            if (fg.propGeoms) {   // step 18: per-geometry transforms for the instanced props
                static std::atomic<uint64_t> pf{0};
                const uint64_t n = ++pf;
                if (n == 1 || n % 600 == 0) Logf("rt: instanced props in the dynamic BLAS: %u instances, %llu tris (+ %llu tris of characters)", fg.propGeoms, (unsigned long long)fg.propTris, (unsigned long long)fg.dynTris);
            }
            if (fg.dynXf.size() == fg.dynGeoms.size() && fg.dynGeoms.size() <= kXfCap && ds->xfPtr) {
                memcpy(ds->xfPtr, fg.dynXf.data(), fg.dynXf.size() * sizeof(VkTransformMatrixKHR));
                for (size_t i = 0; i < fg.dynGeoms.size(); ++i)
                    fg.dynGeoms[i].geometry.triangles.transformData.deviceAddress = fg.dynProp[i] ? ds->xfAddr + (VkDeviceAddress)i * 48 : 0;
            }
            VkAccelerationStructureBuildGeometryInfoKHR bi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
            bi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
            bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            bi.geometryCount = (uint32_t)fg.dynGeoms.size();
            bi.pGeometries = fg.dynGeoms.data();
            std::vector<uint32_t> mp(fg.dynGeoms.size());
            for (size_t i = 0; i < mp.size(); ++i) mp[i] = fg.dynRanges[i].primitiveCount;
            VkAccelerationStructureBuildSizesInfoKHR dsz{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
            d->GetAccelerationStructureBuildSizesKHR(d->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, mp.data(), &dsz);
            if (dsz.accelerationStructureSize > kDynBlasCap || dsz.buildScratchSize > kDynScratchCap) why = "BLAS bigger than the preallocated buffers";
            else {
                VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
                mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_SHADER_READ_BIT;
                d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &mb, 0, nullptr, 0, nullptr);
                const VkDeviceAddress scr = (ds->scratchAddr + d->asScratchAlign - 1) & ~(d->asScratchAlign - 1);
                bi.dstAccelerationStructure = ds->blas;
                bi.scratchData.deviceAddress = scr;
                const VkAccelerationStructureBuildRangeInfoKHR* pR = fg.dynRanges.data();
                d->CmdBuildAccelerationStructuresKHR(cb, 1, &bi, &pR);
                VkMemoryBarrier asb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                asb.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
                asb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
                d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &asb, 0, nullptr, 0, nullptr);
                VkAccelerationStructureInstanceKHR inst[2] = {};
                for (int k = 0; k < 2; ++k) {
                    inst[k].transform.matrix[0][0] = inst[k].transform.matrix[1][1] = inst[k].transform.matrix[2][2] = 1.0f;
                    inst[k].mask = 0xFF;
                    inst[k].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
                }
                inst[0].accelerationStructureReference = slot->blasAddr;
                inst[1].accelerationStructureReference = ds->blasAddr;
                memcpy(ds->instPtr, inst, sizeof inst);
                VkAccelerationStructureGeometryKHR tg{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
                tg.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
                tg.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
                tg.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
                tg.geometry.instances.arrayOfPointers = VK_FALSE;
                tg.geometry.instances.data.deviceAddress = ds->instAddr;
                VkAccelerationStructureBuildGeometryInfoKHR tb{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
                tb.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
                tb.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
                tb.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
                tb.geometryCount = 1;
                tb.pGeometries = &tg;
                tb.dstAccelerationStructure = ds->tlas;
                tb.scratchData.deviceAddress = scr;
                VkAccelerationStructureBuildRangeInfoKHR trange{};
                trange.primitiveCount = 2;
                const VkAccelerationStructureBuildRangeInfoKHR* pTr = &trange;
                d->CmdBuildAccelerationStructuresKHR(cb, 1, &tb, &pTr);
                {
                    std::lock_guard<std::mutex> lk(d->cacheMtx);
                    ds->lastUse = d->frames.load();
                }
                traceTlas = ds->tlas;
            }
        }
        if (why) {
            d->dynSkipped++;
            static std::atomic<int> dsl{0};
            if (dsl.fetch_add(1) < 8) Logf("rt: dynamic geometry skipped this frame: %s (%zu geoms, %llu tris)", why, fg.dynGeoms.size(), (unsigned long long)fg.dynTris);
        }
    }
    {   // AS build (this or an earlier submission) -> ray queries
        VkMemoryBarrier tbar{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        tbar.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        tbar.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT;
        d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                              &tbar, 0, nullptr, 0, nullptr);
    }

    // ---- result images, parameters, descriptors for this frame
    VkImageView giV = VK_NULL_HANDLE, shV = VK_NULL_HANDLE;
    VkImage giI = VK_NULL_HANDLE, shI = VK_NULL_HANDLE;
    {
        std::lock_guard<std::mutex> lk(d->paintMtx);
        if (!EnsureFxImages(d, m.w, m.h)) return;
        giV = d->giView; shV = d->shView; giI = d->giImg; shI = d->shImg;
    }
    const bool taaOn = g_taa && d->hView[0] && d->hView[1] && d->hView[2] && d->hView[3];
    const uint32_t hc = d->histCur & 1u, hp = hc ^ 1u;   // write set / read set (A = index 0,1; B = index 2,3)
    const bool useHist = taaOn && d->histValid;

    FxParams* par = (FxParams*)d->parPtr[ringIdx];
    memcpy(par->vp, vp, sizeof vp);
    memcpy(par->inv, inv, sizeof inv);
    const double az = g_sunAz * 3.14159265358979 / 180.0, el = g_sunEl * 3.14159265358979 / 180.0;
    float sunv[3] = {(float)(cos(el) * sin(az)), (float)sin(el), (float)(cos(el) * cos(az))};
    if (g_autoSun) {  // the game's light direction (travel direction, offset 560 of the frame UBO) -> direction TO the sun
        float ld[3] = {};
        if (ReadBufHost(d, ubo.buf, ubo.off + 560, ld, sizeof ld) && FiniteMat(ld, 3)) {
            const float len = sqrtf(ld[0] * ld[0] + ld[1] * ld[1] + ld[2] * ld[2]);
            if (len > 0.9f && len < 1.1f && ld[1] < 0.0f) {
                sunv[0] = -ld[0] / len; sunv[1] = -ld[1] / len; sunv[2] = -ld[2] / len;
                {   // step 19: the UBO at +560 flips between unrelated values within a few frames -> reject short outliers, accept a new direction only when it persists, then ease to it
                    static float cur[3] = {0, 0, 0}, cand[3] = {0, 0, 0};
                    static int candN = 0;
                    static bool have = false;
                    static std::mutex smx;
                    std::lock_guard<std::mutex> sl(smx);
                    if (!have) { memcpy(cur, sunv, sizeof cur); have = true; }
                    else {
                        auto dist = [](const float* x, const float* y) { return fabsf(x[0] - y[0]) + fabsf(x[1] - y[1]) + fabsf(x[2] - y[2]); };
                        if (dist(sunv, cur) < 0.12f) { candN = 0; for (int i = 0; i < 3; ++i) cur[i] += (sunv[i] - cur[i]) * 0.1f; }
                        else if (candN > 0 && dist(sunv, cand) < 0.12f) {
                            if (++candN >= 45) { memcpy(cur, cand, sizeof cur); candN = 0; }   // persisted ~45 samples: a real scene change
                            else for (int i = 0; i < 3; ++i) cand[i] += (sunv[i] - cand[i]) * 0.2f;
                        } else { memcpy(cand, sunv, sizeof cand); candN = 1; }
                    }
                    const float l2 = sqrtf(cur[0] * cur[0] + cur[1] * cur[1] + cur[2] * cur[2]);
                    if (l2 > 0.5f) { sunv[0] = cur[0] / l2; sunv[1] = cur[1] / l2; sunv[2] = cur[2] / l2; }
                }
                static float lastLog[3] = {9, 9, 9};
                if (fabsf(lastLog[0] - sunv[0]) + fabsf(lastLog[1] - sunv[1]) + fabsf(lastLog[2] - sunv[2]) > 0.08f) {
                    static std::atomic<int> sunLogs{0};
                    if (sunLogs.fetch_add(1) < 25) Logf("rt: sun direction from the game (UBO+560): to-sun = %.3f %.3f %.3f", sunv[0], sunv[1], sunv[2]);
                    memcpy(lastLog, sunv, sizeof lastLog);
                }
            }
        }
    }
    par->sun[0] = sunv[0]; par->sun[1] = sunv[1]; par->sun[2] = sunv[2]; par->sun[3] = g_strength;
    par->cam[0] = cam[0]; par->cam[1] = cam[1]; par->cam[2] = cam[2]; par->cam[3] = g_flipY ? 1.0f : 0.0f;
    const bool lightsOn = g_lightOn && d->lightPipe && d->lightBuf && g_view == 0;
    uint32_t modeEff = 0, flagsEff = (g_shadows ? 1u : 0u) | (g_aoOn ? 2u : 0u) | (g_giOn ? 4u : 0u);
    switch (g_view) {
        case 1: flagsEff = 1u; break;
        case 2: flagsEff = 2u; break;
        case 3: flagsEff = 4u; break;
        case 4: modeEff = 4u; flagsEff = 2u; break;
        case 5: modeEff = 5u; flagsEff = 4u; break;
        case 6: modeEff = 1u; break;
        case 7: modeEff = 2u; break;
        default: flagsEff |= 8u; if (lightsOn) flagsEff |= 16u; break;
    }
    uint32_t lightTile = 48;
    while (((m.w + lightTile - 1) / lightTile) * ((m.h + lightTile - 1) / lightTile) > 4096u) lightTile += 16;
    const uint32_t lightTx = (m.w + lightTile - 1) / lightTile, lightTy = (m.h + lightTile - 1) / lightTile;
    par->misc[0] = m.w; par->misc[1] = m.h; par->misc[2] = modeEff; par->misc[3] = (uint32_t)d->paintInjected.load();
    par->flags[0] = flagsEff;
    par->flags[1] = (uint32_t)g_shRays;
    par->flags[2] = (uint32_t)g_aoRays;
    par->flags[3] = g_reflOn ? 1u : 0u;
    par->fx[0] = g_aoStrength; par->fx[1] = g_aoRadius; par->fx[2] = g_giStrength; par->fx[3] = g_giRange;
    memcpy(par->pvp, d->prevVp, sizeof par->pvp);
    par->tp[0] = useHist ? 1.0f : 0.0f; par->tp[1] = g_taaN; par->tp[2] = 0.0f; par->tp[3] = 0.0f;
    par->lt[0] = g_lightStrength; par->lt[1] = g_lightRange; par->lt[2] = g_lightThr; par->lt[3] = (flagsEff & 4u) ? g_giStrength * 0.5f : 0.0f;
    par->lm[0] = lightTile; par->lm[1] = lightTx; par->lm[2] = (uint32_t)g_lightRays; par->lm[3] = lightsOn ? (1u | (g_lightDebug ? 2u : 0u)) : 0u;
    par->fx2[0] = tanf(g_sunSize * 3.14159265f / 180.0f); par->fx2[1] = g_reflStrength; par->fx2[2] = (float)(d->paintInjected.load() % 100000) * 0.0133f; par->fx2[3] = g_glint;

    {
        std::lock_guard<std::mutex> lk(d->paintMtx);
        VkDescriptorImageInfo ii[8] = {};
        ii[0].imageView = cv;  ii[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ii[1].imageView = dv;  ii[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ii[2].imageView = giV; ii[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ii[3].imageView = shV; ii[3].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSetAccelerationStructureKHR asw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
        asw.accelerationStructureCount = 1;
        asw.pAccelerationStructures = &traceTlas;
        // history: bindings 6/7 = this frame (write), 8/9 = previous frame (read); without history images the main views are bound as dummies
        ii[4].imageView = taaOn ? d->hView[hc] : giV;      ii[4].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ii[5].imageView = taaOn ? d->hView[2 + hc] : shV;  ii[5].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ii[6].imageView = taaOn ? d->hView[hp] : giV;      ii[6].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ii[7].imageView = taaOn ? d->hView[2 + hp] : shV;  ii[7].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet w[9] = {};
        const uint32_t bind[9] = {0, 1, 3, 4, 6, 7, 8, 9, 2};
        const VkDescriptorType wt[9] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR};
        for (int i = 0; i < 9; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = set;
            w[i].dstBinding = bind[i];
            w[i].descriptorCount = 1;
            w[i].descriptorType = wt[i];
            if (i < 8) w[i].pImageInfo = &ii[i]; else w[i].pNext = &asw;
        }
        d->UpdateDescriptorSets(d->device, 9, w, 0, nullptr);
    }

    // the render pass left the color image in SHADER_READ_ONLY and the depth image in GENERAL (see the log)
    VkImageMemoryBarrier pre[4] = {};
    for (int i = 0; i < 4; ++i) {
        pre[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        pre[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        pre[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        pre[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    pre[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    pre[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    pre[0].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    pre[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    pre[0].image = m.color;
    pre[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    pre[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    pre[1].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    pre[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    pre[1].image = m.depth;
    pre[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    pre[2].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;  // previous frame's composite read it
    pre[2].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    pre[2].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;      // contents are fully rewritten every frame
    pre[2].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    pre[2].image = giI;
    pre[3] = pre[2];
    pre[3].image = shI;
    d->CmdPipelineBarrier(cb,
                          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                              VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 4, pre);

    // ---- pass 0 (step 21): find emissive light sources. The list is cleared, filled tile by tile and then read by the trace pass
    if (lightsOn) {
        VkBufferMemoryBarrier bb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb.buffer = d->lightBuf;
        bb.offset = 0;
        bb.size = VK_WHOLE_SIZE;
        bb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;   // previous frame
        bb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &bb, 0, nullptr);
        d->CmdFillBuffer(cb, d->lightBuf, 0, 16, 0);
        bb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        bb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &bb, 0, nullptr);
        d->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d->lightPipe);
        d->CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d->paintPl, 0, 1, &set, 0, nullptr);
        d->CmdDispatch(cb, lightTx, lightTy, 1);
        bb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &bb, 0, nullptr);
        if (d->statBuf) { VkBufferCopy rg{0, 0, 16}; d->CmdCopyBuffer(cb, d->lightBuf, d->statBuf, 1, &rg); }
    }

    // ---- pass 1: trace
    d->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d->paintPipe);  // original function: not our graphics hook
    d->CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d->paintPl, 0, 1, &set, 0, nullptr);
    d->CmdDispatch(cb, (m.w + 7) / 8, (m.h + 7) / 8, 1);

    VkImageMemoryBarrier mid[2] = {};
    for (int i = 0; i < 2; ++i) {
        mid[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        mid[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mid[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        mid[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        mid[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        mid[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        mid[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        mid[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    mid[0].image = giI;
    mid[1].image = shI;
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
                          2, mid);

    if (taaOn) {   // history: previous result readable, this frame's set writable (old contents are not needed)
        VkImageMemoryBarrier hb[4] = {};
        for (int i = 0; i < 4; ++i) {
            hb[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            hb[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            hb[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            hb[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            hb[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        }
        for (int k = 0; k < 2; ++k) {   // write set (A and B)
            VkImageMemoryBarrier& b2 = hb[k];
            b2.image = d->hImg[k == 0 ? hc : 2 + hc];
            b2.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            b2.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
            b2.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        }
        for (int k = 0; k < 2; ++k) {   // read set (A and B)
            VkImageMemoryBarrier& b2 = hb[2 + k];
            b2.image = d->hImg[k == 0 ? hp : 2 + hp];
            b2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            b2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            b2.oldLayout = d->histValid ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;
        }
        d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 4, hb);
    }
    // ---- pass 2: denoise + composite into the game's color image
    d->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d->fxPipe);
    d->CmdDispatch(cb, (m.w + 7) / 8, (m.h + 7) / 8, 1);

    VkImageMemoryBarrier post = {};
    post.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    post.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    post.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    post.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    post.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    post.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    post.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    post.image = m.color;
    post.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                          nullptr, 1, &post);

    if (taaOn) {   // this frame's output becomes the next frame's history
        memcpy(d->prevVp, vp, sizeof d->prevVp);
        d->histCur ^= 1u;
        d->histValid = true;
    } else d->histValid = false;
    d->rtLastGeoms = geoms.empty() ? d->rtLastGeoms.load() : geoms.size();
    d->rtLastTris = cacheTrisNow;
    if (rebuild) d->rtLastAsKB = sz.accelerationStructureSize >> 10;
    const uint64_t n = ++d->paintInjected;
    d->rtBuilt++;
    if (lightsOn && d->statPtr && (n % 600 == 7)) {   // the header of the light list from an earlier frame (read without sync: only used for the log)
        uint32_t hdr[4] = {};
        memcpy(hdr, d->statPtr, sizeof hdr);
        Logf("rt: emissive lights: %u light sources found%s | brightest pixel %.2f, threshold lightthr=%.2f (raise it if ordinary surfaces glow, lower it if fire does not light anything)",
             std::min<uint32_t>(hdr[0], kLightMax), hdr[0] > kLightMax ? " (more than fit - raise lightthr)" : "", hdr[1] / 1000.0, (double)g_lightThr);
    }
    if (n == 1 || n == 2000) {
        Logf("rt: %s frame: drew %zu geometries, %llu triangles (draws %llu; skipped: pipeline/format %llu, range %llu, other %llu, "
             "unreadable %llu) | cache holds %llu triangles",
             n == 1 ? "first" : "frame 2000", fg.geoms.size(), (unsigned long long)fg.tris, (unsigned long long)fg.draws,
             (unsigned long long)fg.skipPipe, (unsigned long long)fg.skipRange, (unsigned long long)fg.skipOther,
             (unsigned long long)fg.unreadable, (unsigned long long)cacheTrisNow);
        Logf("rt: camera %.3f %.3f %.3f | frame UBO %p off=%llu (%s) | sun az=%.1f el=%.1f strength=%.2f view=%u flipY=%d", cam[0],
             cam[1], cam[2], (void*)ubo.buf, (unsigned long long)ubo.off, haveUboBound ? "bound in the main pass" : "fallback cand 0",
             g_sunAz, g_sunEl, (double)g_strength, g_view, g_flipY ? 1 : 0);
        DumpUnitVectors(d, ubo);
        Logf("rt: controls: Ctrl+Home = ray tracing on/off, Ctrl+End = next view; everything else in SkyRT.cfg next to the DLL (applied live)");
    }
}

// called after the original vkCmdEndRenderPass
static void MainPassEnd(DeviceData* d, VkCommandBuffer cb) {
    DeviceData::MainCb m{};
    {
        std::lock_guard<std::mutex> lk(d->paintMtx);
        auto it = d->mainCb.find(cb);
        if (it == d->mainCb.end()) return;
        m = it->second;
        d->mainCb.erase(it);
        d->mainCount--;
    }
    LARGE_INTEGER t0, t1, fq;
    QueryPerformanceFrequency(&fq);
    QueryPerformanceCounter(&t0);
    InjectPaint(d, cb, m);
    QueryPerformanceCounter(&t1);
    const uint64_t us = (uint64_t)((t1.QuadPart - t0.QuadPart) * 1000000ll / fq.QuadPart);
    d->usInject += us;
    d->injectN++;
    uint64_t mx = d->usInjectMax.load();
    while (us > mx && !d->usInjectMax.compare_exchange_weak(mx, us)) {}
    if (us > 3000 && d->spikes.fetch_add(1) < 40) Logf("rt: CPU spike: recording the RT pass took %.2f ms (rebuilds so far %llu)", us / 1000.0, (unsigned long long)d->rtRebuilds.load());
}

static VKAPI_ATTR void VKAPI_CALL SkyRT_DestroyImage(VkDevice device, VkImage image, const VkAllocationCallbacks* pAllocator) {
    DeviceData* d = GetDev(Key(device));
    if (!d) return;
    {
        std::lock_guard<std::mutex> lk(d->regMtx);
        d->imgs.erase(image);
    }
    VkImageView own = VK_NULL_HANDLE;
    {
        std::lock_guard<std::mutex> lk(d->paintMtx);
        auto it = d->ownViews.find(image);
        if (it != d->ownViews.end()) { own = it->second; d->ownViews.erase(it); }
    }
    if (own) d->DestroyImageView(device, own, nullptr);
    d->DestroyImage(device, image, pAllocator);
}

// finds the descriptor that points at the global frame UBO (the struct the vertex shader reads is 1584 bytes)
static VKAPI_ATTR void VKAPI_CALL SkyRT_UpdateDescriptorSets(VkDevice device, uint32_t writeCount,
                                                             const VkWriteDescriptorSet* pWrites, uint32_t copyCount,
                                                             const VkCopyDescriptorSet* pCopies) {
    DeviceData* d = GetDev(Key(device));
    if (!d) return;
    if (pWrites && (d->track || !d->uboScanDone.load(std::memory_order_relaxed))) {
        for (uint32_t i = 0; i < writeCount; ++i) {
            const VkWriteDescriptorSet& w = pWrites[i];
            if ((w.descriptorType != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER &&
                 w.descriptorType != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC) || !w.pBufferInfo)
                continue;
            for (uint32_t k = 0; k < w.descriptorCount; ++k) {
                const VkDescriptorBufferInfo& bi = w.pBufferInfo[k];
                if (bi.range != 1584) continue;
                d->uboWrites++;
                std::lock_guard<std::mutex> lk(d->uboMtx);
                if (w.dstSet && d->setUbo.size() < 200000) d->setUbo[w.dstSet] = DeviceData::UboRef{bi.buffer, bi.offset};
                bool have = false;
                for (const DeviceData::UboCand& c : d->uboCands)
                    if (c.buf == bi.buffer && c.off == bi.offset) { have = true; break; }
                if (have || d->uboScanDone.load(std::memory_order_relaxed)) continue;
                if (d->uboCands.size() < 64) d->uboCands.push_back(DeviceData::UboCand{bi.buffer, bi.offset, bi.range, (uint32_t)w.descriptorType});
                else d->uboScanDone = true;
            }
        }
    }
    d->UpdateDescriptorSets(device, writeCount, pWrites, copyCount, pCopies);
}

static VKAPI_ATTR void VKAPI_CALL SkyRT_CmdBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bindPoint,
                                                              VkPipelineLayout layout, uint32_t firstSet, uint32_t setCount,
                                                              const VkDescriptorSet* pSets, uint32_t dynCount,
                                                              const uint32_t* pDyn) {
    DeviceData* d = GetDev(Key(cb));
    if (!d) return;
    if (dynCount && d->capturing.load(std::memory_order_relaxed)) {
        d->dynBinds++;
        std::lock_guard<std::mutex> lk(d->uboMtx);
        for (uint32_t i = 0; i < dynCount && d->dynOffsets.size() < 64; ++i) d->dynOffsets.insert(pDyn[i]);
    }
    if (d->track && d->mainCount.load(std::memory_order_relaxed) > 0 && bindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS) {
        bool inMain;
        { std::lock_guard<std::mutex> lk(d->paintMtx); inMain = d->mainCb.count(cb) != 0; }
        if (inMain) {
            std::lock_guard<std::mutex> lk(d->uboMtx);
            for (uint32_t i = 0; i < setCount; ++i) {
                auto it = d->setUbo.find(pSets[i]);
                if (it != d->setUbo.end()) { d->mainUbo[cb] = it->second; break; }
            }
        }
    }
    d->CmdBindDescriptorSets(cb, bindPoint, layout, firstSet, setCount, pSets, dynCount, pDyn);
}

static void LogStep5Info(DeviceData* d) {
    Logf("  indirect args read at record time vs at submit time: checked %llu, differing %llu, not readable at record time %llu",
         (unsigned long long)d->recChecked.load(), (unsigned long long)d->recDiffer.load(),
         (unsigned long long)d->recUnreadable.load());
    {
        std::lock_guard<std::mutex> lk(d->uboMtx);
        if (d->dynOffsets.empty())
            Logf("  vkCmdBindDescriptorSets with dynamic offsets: %llu binds", (unsigned long long)d->dynBinds.load());
        else
            Logf("  vkCmdBindDescriptorSets with dynamic offsets: %llu binds, %zu distinct offsets (min %u, max %u)",
                 (unsigned long long)d->dynBinds.load(), d->dynOffsets.size(), (unsigned)*d->dynOffsets.begin(),
                 (unsigned)*d->dynOffsets.rbegin());
    }
    std::vector<DeviceData::UboCand> cands;
    {
        std::lock_guard<std::mutex> lk(d->uboMtx);
        cands = d->uboCands;
    }
    Logf("  global frame UBO candidates (descriptor range == 1584): %zu, matching descriptor writes so far: %llu",
         cands.size(), (unsigned long long)d->uboWrites.load());
    for (size_t i = 0; i < cands.size() && i < 6; ++i) {
        float m[16] = {};
        float cam[3] = {};
        const bool ok1 = ReadBufHost(d, cands[i].buf, cands[i].off + 224, m, sizeof m);
        const bool ok2 = ReadBufHost(d, cands[i].buf, cands[i].off + 608, cam, sizeof cam);
        Logf("    ubo %p off=%llu range=%llu type=%s host-readable=%d", (void*)cands[i].buf,
             (unsigned long long)cands[i].off, (unsigned long long)cands[i].range,
             cands[i].type == (uint32_t)VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ? "DYNAMIC" : "STATIC",
             (ok1 && ok2) ? 1 : 0);
        if (ok1 && ok2)
            Logf("      child8 columns: [%.4f %.4f %.4f %.4f] [%.4f %.4f %.4f %.4f] [%.4f %.4f %.4f %.4f] [%.4f %.4f %.4f %.4f] | "
                 "child17: %.3f %.3f %.3f",
                 m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15],
                 cam[0], cam[1], cam[2]);
    }
}

static void ArmCapture(DeviceData* d) {
    {
        std::lock_guard<std::mutex> lk(d->cmdMtx);
        d->resolved.clear();
        d->groups.clear();
        d->passes.clear();
        d->unreadable = 0;
        d->cmdState.clear();
        d->cmdDraws.clear();
    }
    d->recChecked = 0;
    d->recDiffer = 0;
    d->recUnreadable = 0;
    d->dynBinds = 0;
    {
        std::lock_guard<std::mutex> lk(d->uboMtx);
        d->dynOffsets.clear();
    }
    d->capturing = true;
    Logf("frame capture armed: recording the next frame");
}

static void FinishCapture(DeviceData* d) {
    d->capturing = false;
    g_stage = "FinishCapture";
    std::vector<DeviceData::Resolved> res;
    std::vector<DeviceData::Group> groups;
    uint64_t unreadable = 0;
    {
        std::lock_guard<std::mutex> lk(d->cmdMtx);
        res.swap(d->resolved);
        groups.swap(d->groups);
        unreadable = d->unreadable;
        d->unreadable = 0;
        d->cmdState.clear();
        d->cmdDraws.clear();
    }
    uint64_t direct = 0, viaInd = 0, tris = 0, trisInd = 0;
    uint64_t bucket[5] = {0, 0, 0, 0, 0};
    std::unordered_map<VkBuffer, uint32_t> ibUse, vbUse;
    for (const DeviceData::Resolved& r : res) {
        const uint64_t t = (uint64_t)(r.c.indexCount / 3) * r.c.instanceCount;
        tris += t;
        if (r.fromIndirect) { viaInd++; trisInd += t; } else { direct++; }
        const uint32_t n = r.c.indexCount;
        bucket[n <= 36 ? 0 : n <= 300 ? 1 : n <= 1000 ? 2 : n <= 5000 ? 3 : 4]++;
        if (r.st.ib) ibUse[r.st.ib]++;
        for (int s = 0; s < 8; ++s)
            if (r.st.vb[s]) vbUse[r.st.vb[s]]++;
    }
    Logf("--- frame capture --- draws: %llu direct + %llu from indirect, triangles: %llu total (%llu via indirect), "
         "indirect calls: %zu (unreadable: %llu)",
         (unsigned long long)direct, (unsigned long long)viaInd, (unsigned long long)tris, (unsigned long long)trisInd,
         groups.size(), (unsigned long long)unreadable);
    Logf("  indexCount buckets: <=36: %llu, <=300: %llu, <=1000: %llu, <=5000: %llu, >5000: %llu",
         (unsigned long long)bucket[0], (unsigned long long)bucket[1], (unsigned long long)bucket[2],
         (unsigned long long)bucket[3], (unsigned long long)bucket[4]);
    LogTop(d, "index buffers", ibUse, 6);
    LogTop(d, "vertex buffers (all slots)", vbUse, 6);

    // vertex layout (stride / rate / attribute formats) of every pipeline that drew in this frame
    std::unordered_map<VkPipeline, std::pair<uint32_t, uint32_t>> pipeUse;  // draws, of them via indirect
    for (const DeviceData::Resolved& r : res) {
        std::pair<uint32_t, uint32_t>& u = pipeUse[r.st.pipe];
        u.first++;
        if (r.fromIndirect) u.second++;
    }
    Logf("  pipelines used: %zu", pipeUse.size());
    size_t pshown = 0;
    for (const auto& kv : pipeUse) {
        if (pshown++ >= 16) break;
        DeviceData::PipeInfo pi;
        bool known = false;
        {
            std::lock_guard<std::mutex> lk(d->regMtx);
            auto it = d->pipes.find(kv.first);
            if (it != d->pipes.end()) { pi = it->second; known = true; }
        }
        char line[760];
        size_t p = (size_t)snprintf(line, sizeof line, "  pipeline %p: %u draws (%u via indirect)%s", (void*)kv.first,
                                    kv.second.first, kv.second.second, known ? " | bindings:" : " | no vertex info");
        if (known) {
            for (const DeviceData::PipeInfo::Bind& b : pi.binds)
                if (p + 90 < sizeof line)
                    p += (size_t)snprintf(line + p, sizeof line - p, " [b%u stride=%u %s]", b.binding, b.stride,
                                          b.rate ? "INSTANCE" : "VERTEX");
            if (p + 12 < sizeof line) p += (size_t)snprintf(line + p, sizeof line - p, " | attrs:");
            for (const DeviceData::PipeInfo::Attr& a : pi.attrs)
                if (p + 90 < sizeof line)
                    p += (size_t)snprintf(line + p, sizeof line - p, " loc%u->b%u@%u %s", a.loc, a.binding, a.offset,
                                          FormatName(a.format));
        }
        Logf("%s", line);
    }

    size_t shown = 0;
    for (const DeviceData::Group& g : groups) {
        if (shown++ >= 14) break;
        char vbs[200] = {};
        size_t pos = 0;
        for (int s = 0; s < 8 && pos + 60 < sizeof vbs; ++s)
            if (g.st.vb[s])
                pos += (size_t)snprintf(vbs + pos, sizeof vbs - pos, " vb[%d]=%p(%llu)", s, (void*)g.st.vb[s],
                                        BufSizeOf(d, g.st.vb[s]));
        Logf("  indirect call: off=%llu count=%u stride=%u readable=%d ib=%p(%llu) type=%d pipe=%p%s",
             (unsigned long long)g.indOff, g.drawCount, g.stride, g.readable ? 1 : 0, (void*)g.st.ib,
             BufSizeOf(d, g.st.ib), (int)g.st.ibType, (void*)g.st.pipe, vbs);
    }
    // render passes of the captured frame: which images each one draws to (finds the main HDR + depth pass)
    std::vector<DeviceData::PassEv> passes;
    {
        std::lock_guard<std::mutex> lk(d->cmdMtx);
        passes.swap(d->passes);
    }
    Logf("  render passes in the captured frame: %zu | attachment usage patches: ok=%llu failed=%llu", passes.size(),
         (unsigned long long)d->imgPatched.load(), (unsigned long long)d->imgPatchFail.load());
    size_t pidx = 0;
    for (const DeviceData::PassEv& pe : passes) {
        if (pidx >= 60) break;
        char pl[900];
        size_t pos2 = (size_t)snprintf(pl, sizeof pl, "  pass %zu: rp=%p fb=%p area=%ux%u draw calls=%zu |", pidx, (void*)pe.rp,
                                       (void*)pe.fb, pe.w, pe.h, pe.drawEnd - pe.drawStart);
        {
            std::lock_guard<std::mutex> lk(d->regMtx);
            auto fit = d->fbs.find(pe.fb);
            auto rit = d->rps.find(pe.rp);
            if (fit == d->fbs.end()) {
                pos2 += (size_t)snprintf(pl + pos2, sizeof pl - pos2, " no tracked attachment images (swapchain or small)");
            } else {
                for (size_t i = 0; i < fit->second.views.size() && pos2 + 200 < sizeof pl; ++i) {
                    auto vit = d->views.find(fit->second.views[i]);
                    auto iit = (vit == d->views.end()) ? d->imgs.end() : d->imgs.find(vit->second.image);
                    if (iit == d->imgs.end()) {
                        pos2 += (size_t)snprintf(pl + pos2, sizeof pl - pos2, " [att%zu untracked]", i);
                        continue;
                    }
                    const DeviceData::ImgInfo& ii = iit->second;
                    pos2 += (size_t)snprintf(pl + pos2, sizeof pl - pos2, " [att%zu img=%p %s %ux%u usage=0x%x", i,
                                             (void*)vit->second.image, FormatName((uint32_t)ii.format), ii.w, ii.h,
                                             (unsigned)ii.patchedUsage);
                    if (rit != d->rps.end() && i < rit->second.size()) {
                        char fl[40];
                        LayoutStr((int)rit->second[i].finalLayout, fl, sizeof fl);
                        pos2 += (size_t)snprintf(pl + pos2, sizeof pl - pos2, " load=%s final=%s",
                                                 LoadOpStr((int)rit->second[i].loadOp), fl);
                    }
                    pos2 += (size_t)snprintf(pl + pos2, sizeof pl - pos2, "]");
                }
            }
            if (rit != d->rps.end() && rit->second.size() == 2 &&
                rit->second[0].format == (uint32_t)VK_FORMAT_R16G16B16A16_SFLOAT && rit->second[0].loadOp == 1 &&
                rit->second[1].format == (uint32_t)VK_FORMAT_D32_SFLOAT && rit->second[1].loadOp == 1 && pos2 + 40 < sizeof pl)
                pos2 += (size_t)snprintf(pl + pos2, sizeof pl - pos2, " <== MAIN CANDIDATE");
        }
        Logf("%s", pl);
        pidx++;
    }
    Logf("  render pass begins so far: vkCmdBeginRenderPass=%llu vkCmdBeginRenderPass2=%llu",
         (unsigned long long)d->rpBegin1.load(), (unsigned long long)d->rpBegin2.load());
    LogStep5Info(d);
    Logf("--- end of capture ---");
    BuildWorldBlasTest(d, res);
}

// ============================================================ frame / submit / draws
static VKAPI_ATTR VkResult VKAPI_CALL SkyRT_QueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* pPresentInfo) {
    DeviceData* d = GetDev(Key(queue));
    if (!d) return VK_ERROR_INITIALIZATION_FAILED;
    g_stage = "vkQueuePresentKHR";
    VkResult r = d->QueuePresentKHR(queue, pPresentInfo);
    uint64_t f = ++d->frames;
    d->presentQueue = queue;
    {   // step 14: frame-time hitch detector (what happened around a frame that took much longer than usual)
        static LARGE_INTEGER fq = {}, last = {};
        static double avg = 16.7;
        LARGE_INTEGER now; QueryPerformanceCounter(&now);
        if (!fq.QuadPart) QueryPerformanceFrequency(&fq);
        if (last.QuadPart) {
            const double ms = (double)(now.QuadPart - last.QuadPart) * 1000.0 / (double)fq.QuadPart;
            if (f > 120 && ms > 28.0 && ms > avg * 2.0 && d->hitches.fetch_add(1) < 60) {
                const uint64_t lr = d->lastRebuildFrameA.load();
                Logf("rt: HITCH frame %llu took %.1f ms (avg %.1f) | last BLAS rebuild %llu frames ago | cache %llu tris | rebuilds %llu",
                     (unsigned long long)f, ms, avg, (unsigned long long)(f - lr), (unsigned long long)d->rtLastTris.load(),
                     (unsigned long long)d->rtRebuilds.load());
            }
            if (ms < 100.0) avg = avg * 0.97 + ms * 0.03;
        }
        last = now;
    }
    static std::atomic<int> errLogs{0};
    if (r < 0 && errLogs.fetch_add(1) < 20) {
        Logf("!!! vkQueuePresentKHR returned %d%s", (int)r, r == VK_ERROR_DEVICE_LOST ? " (DEVICE_LOST)" : "");
        FlushToDisk();
    }
    {  // release game buffers / memory whose destruction was held back (no frame in flight can use them now)
        std::vector<DeviceData::Deferred> rel;
        {
            std::lock_guard<std::mutex> lk(d->cacheMtx);
            for (size_t i = 0; i < d->deferred.size();) {
                if (f - d->deferred[i].frame >= 12) { rel.push_back(d->deferred[i]); d->deferred[i] = d->deferred.back(); d->deferred.pop_back(); }
                else ++i;
            }
        }
        for (auto& e : rel) {
            if (e.buf) d->DestroyBuffer(d->device, e.buf, nullptr);
            if (e.mem) d->FreeMemory(d->device, e.mem, nullptr);
        }
    }
    if (d->track) PollKeys(f);
    if (f == 1) Logf("first vkQueuePresentKHR: the game renders through the layer");
    if (f == 1 || f % kStatsEveryNFrames == 0) PrintStats(d, "stats");
    if (f == 300 || f == 2400) ProbeBuffers(d);
    if (d->capturing.load()) FinishCapture(d);                 // the frame recorded since the previous present
    if (f == 600 || f == 2400 || f == 6000) ArmCapture(d);     // record the next frame
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL SkyRT_QueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo* pSubmits,
                                                        VkFence fence) {
    DeviceData* d = GetDev(Key(queue));
    if (!d) return VK_ERROR_INITIALIZATION_FAILED;
    g_stage = "vkQueueSubmit";
    if (d->capturing.load(std::memory_order_relaxed)) ResolveSubmit(d, submitCount, pSubmits);
    VkResult r = d->QueueSubmit(queue, submitCount, pSubmits, fence);
    d->submits++;
    static std::atomic<int> errLogs{0};
    if (r < 0 && errLogs.fetch_add(1) < 20) {
        Logf("!!! vkQueueSubmit returned %d%s", (int)r, r == VK_ERROR_DEVICE_LOST ? " (DEVICE_LOST)" : "");
        Logf("    context: frame %llu, rebuilds %llu, evicted %llu, cache %llu tris, last stage '%s'", (unsigned long long)d->frames.load(),
             (unsigned long long)d->rtRebuilds.load(), (unsigned long long)d->rtEvicted.load(), (unsigned long long)d->rtLastTris.load(),
             g_stage.load());
        FlushToDisk();
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL SkyRT_CmdDraw(VkCommandBuffer cb, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex,
                                                uint32_t firstInstance) {
    DeviceData* d = GetDev(Key(cb));
    if (!d) return;
    g_cDraw.fetch_add(1, std::memory_order_relaxed);
    if (g_ppState.load(std::memory_order_relaxed) && PassProbeDraw(d, cb)) return;
    if ((g_probeOn.load(std::memory_order_relaxed) || (g_hide.load(std::memory_order_relaxed) >= 0 || g_hideGroup.load(std::memory_order_relaxed) > 0)) && d->mainCount.load(std::memory_order_relaxed) > 0) {
        bool inMain;
        { std::lock_guard<std::mutex> lk(d->paintMtx); inMain = d->mainCb.count(cb) != 0; }
        if (inMain) {
            DeviceData::CmdState st{};
            bool have = false;
            { std::lock_guard<std::mutex> lk(d->cmdMtx); auto it = d->cmdState.find(cb); if (it != d->cmdState.end()) { st = it->second; have = true; } }
            if (have) {
                std::vector<VkDrawIndexedIndirectCommand> one;
                one.push_back(VkDrawIndexedIndirectCommand{vertexCount, instanceCount, 0, 0, firstInstance});
                ProbeBlended(d, st, one, true);
            }
            if (ShouldHide(d, cb)) return;
        }
    }
    d->CmdDraw(cb, vertexCount, instanceCount, firstVertex, firstInstance);
}

static VKAPI_ATTR void VKAPI_CALL SkyRT_CmdDrawIndexed(VkCommandBuffer cb, uint32_t indexCount, uint32_t instanceCount,
                                                       uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) {
    DeviceData* d = GetDev(Key(cb));
    if (!d) return;
    d->drawIndexed.fetch_add(1, std::memory_order_relaxed);
    if (g_ppState.load(std::memory_order_relaxed) && PassProbeDraw(d, cb)) return;
    if (d->capturing.load(std::memory_order_relaxed)) {
        VkDrawIndexedIndirectCommand c{indexCount, instanceCount, firstIndex, vertexOffset, firstInstance};
        RecordDraw(d, cb, false, VK_NULL_HANDLE, 0, 0, 0, c);
    }
    if (d->mainCount.load(std::memory_order_relaxed) > 0) {
        VkDrawIndexedIndirectCommand c{indexCount, instanceCount, firstIndex, vertexOffset, firstInstance};
        MainDraw(d, cb, false, VK_NULL_HANDLE, 0, 0, 0, &c);
        if (ShouldHide(d, cb)) return;
    }
    d->CmdDrawIndexed(cb, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
}

static VKAPI_ATTR void VKAPI_CALL SkyRT_CmdDrawIndexedIndirect(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset,
                                                               uint32_t drawCount, uint32_t stride) {
    DeviceData* d = GetDev(Key(cb));
    if (!d) return;
    d->indirectCalls.fetch_add(1, std::memory_order_relaxed);
    if (g_ppState.load(std::memory_order_relaxed) && PassProbeDraw(d, cb)) return;
    d->indirectSubdraws.fetch_add(drawCount, std::memory_order_relaxed);
    if (d->capturing.load(std::memory_order_relaxed)) {
        VkDrawIndexedIndirectCommand none{};
        std::vector<VkDrawIndexedIndirectCommand> now;
        const bool readNow = ReadIndirect(d, buffer, offset, drawCount,
                                          stride ? stride : (uint32_t)sizeof(VkDrawIndexedIndirectCommand), now);
        RecordDraw(d, cb, true, buffer, offset, drawCount, stride, none, readNow ? HashCmds(now) : 0, readNow);
    }
    if (d->mainCount.load(std::memory_order_relaxed) > 0) {
        MainDraw(d, cb, true, buffer, offset, drawCount, stride, nullptr);
        if (ShouldHide(d, cb)) return;
    }
    d->CmdDrawIndexedIndirect(cb, buffer, offset, drawCount, stride);
}

// step 12: the draw paths we did not hook before (the water may use one of them)
static bool OtherDraw(DeviceData* d, VkCommandBuffer cb, std::atomic<uint64_t>& ctr, const char* what, uint32_t count) {  // true = skip
    ctr.fetch_add(1, std::memory_order_relaxed);
    if (g_ppState.load(std::memory_order_relaxed) && PassProbeDraw(d, cb)) return true;
    if ((g_probeOn.load(std::memory_order_relaxed) || (g_hide.load(std::memory_order_relaxed) >= 0 || g_hideGroup.load(std::memory_order_relaxed) > 0)) && d->mainCount.load(std::memory_order_relaxed) > 0) {
        bool inMain;
        { std::lock_guard<std::mutex> lk(d->paintMtx); inMain = d->mainCb.count(cb) != 0; }
        if (inMain) {
            DeviceData::CmdState st{};
            bool have = false;
            { std::lock_guard<std::mutex> lk(d->cmdMtx); auto it = d->cmdState.find(cb); if (it != d->cmdState.end()) { st = it->second; have = true; } }
            if (have) {
                std::vector<VkDrawIndexedIndirectCommand> one;
                one.push_back(VkDrawIndexedIndirectCommand{count, 1, 0, 0, 0});
                ProbeBlended(d, st, one, true);
            }
            if (ShouldHide(d, cb)) return true;
        }
    }
    static std::atomic<int> logged{0};
    if (logged.fetch_add(1) < 6) Logf("rt: NOTE game uses %s (count %u)", what, count);
    return false;
}
static VKAPI_ATTR void VKAPI_CALL SkyRT_CmdDrawIndirect(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, uint32_t drawCount, uint32_t stride) {
    DeviceData* d = GetDev(Key(cb));
    if (!d) return;
    if (OtherDraw(d, cb, g_cDrawInd, "vkCmdDrawIndirect", drawCount)) return;
    d->CmdDrawIndirect(cb, buffer, offset, drawCount, stride);
}
static VKAPI_ATTR void VKAPI_CALL SkyRT_CmdDrawIndirectCount(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, VkBuffer cbuf, VkDeviceSize coff, uint32_t maxDraw, uint32_t stride) {
    DeviceData* d = GetDev(Key(cb));
    if (!d) return;
    if (OtherDraw(d, cb, g_cDrawIndCnt, "vkCmdDrawIndirectCount", maxDraw)) return;
    d->CmdDrawIndirectCount(cb, buffer, offset, cbuf, coff, maxDraw, stride);
}
static VKAPI_ATTR void VKAPI_CALL SkyRT_CmdDrawIndexedIndirectCount(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, VkBuffer cbuf, VkDeviceSize coff, uint32_t maxDraw, uint32_t stride) {
    DeviceData* d = GetDev(Key(cb));
    if (!d) return;
    if (OtherDraw(d, cb, g_cDrawIdxIndCnt, "vkCmdDrawIndexedIndirectCount", maxDraw)) return;
    d->CmdDrawIndexedIndirectCount(cb, buffer, offset, cbuf, coff, maxDraw, stride);
}
static VKAPI_ATTR void VKAPI_CALL SkyRT_CmdExecuteCommands(VkCommandBuffer cb, uint32_t n, const VkCommandBuffer* p) {
    DeviceData* d = GetDev(Key(cb));
    if (!d) return;
    g_cExec.fetch_add(1, std::memory_order_relaxed);
    static std::atomic<int> logged{0};
    if (logged.fetch_add(1) < 6) Logf("rt: NOTE game uses vkCmdExecuteCommands (secondary command buffers, %u) - draws inside them are invisible to the probes", n);
    d->CmdExecuteCommands(cb, n, p);
}

// step 13: anisotropic filtering + optional LOD bias on every sampler the game creates
static VKAPI_ATTR VkResult VKAPI_CALL SkyRT_CreateSampler(VkDevice device, const VkSamplerCreateInfo* pCreateInfo,
                                                          const VkAllocationCallbacks* pAllocator, VkSampler* pSampler) {
    DeviceData* d = GetDev(Key(device));
    if (!d) return VK_ERROR_INITIALIZATION_FAILED;
    VkSamplerCreateInfo ci = *pCreateInfo;
    static std::atomic<int> logged{0};
    bool changed = false;
    if (d->anisoOk && g_aniso > 1 && !ci.compareEnable && !ci.unnormalizedCoordinates && ci.minFilter == VK_FILTER_LINEAR && ci.magFilter == VK_FILTER_LINEAR &&
        ci.maxLod > 0.5f) {
        ci.anisotropyEnable = VK_TRUE;
        ci.maxAnisotropy = std::min((float)g_aniso, d->maxAniso);
        changed = true;
    }
    if (g_lodBias != 0.0f && ci.maxLod > 0.5f && !ci.unnormalizedCoordinates && !ci.compareEnable) { ci.mipLodBias += g_lodBias; changed = true; }
    if (changed && logged.fetch_add(1) < 3) Logf("rt: sampler patched: anisotropy %.0f, lod bias %.2f (further samplers are patched silently)", ci.anisotropyEnable ? ci.maxAnisotropy : 0.0f, ci.mipLodBias);
    return d->CreateSampler(device, &ci, pAllocator, pSampler);
}

// ============================================================ proc addr + loader interface
extern "C" __declspec(dllexport) VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL SkyRT_GetDeviceProcAddr(VkDevice device,
                                                                                                   const char* name) {
    DeviceData* d = GetDev(Key(device));
    if (!d) return nullptr;
    if (!strcmp(name, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)&SkyRT_GetDeviceProcAddr;
#define HOOKD(fn) if (!strcmp(name, "vk" #fn) && d->fn) return (PFN_vkVoidFunction)&SkyRT_##fn
    HOOKD(DestroyDevice);
    HOOKD(CreateBuffer);
    HOOKD(AllocateMemory);
    HOOKD(QueuePresentKHR);
    HOOKD(QueueSubmit);
    HOOKD(CmdDraw);
    HOOKD(CmdDrawIndexed);
    HOOKD(CmdDrawIndexedIndirect);
    HOOKD(CmdDrawIndirect);
    HOOKD(CreateSampler);
    HOOKD(CmdExecuteCommands);
    if ((!strcmp(name, "vkCmdDrawIndirectCount") || !strcmp(name, "vkCmdDrawIndirectCountKHR")) && d->CmdDrawIndirectCount) return (PFN_vkVoidFunction)&SkyRT_CmdDrawIndirectCount;
    if ((!strcmp(name, "vkCmdDrawIndexedIndirectCount") || !strcmp(name, "vkCmdDrawIndexedIndirectCountKHR")) && d->CmdDrawIndexedIndirectCount) return (PFN_vkVoidFunction)&SkyRT_CmdDrawIndexedIndirectCount;
    HOOKD(BindBufferMemory);
    HOOKD(BindBufferMemory2);
    HOOKD(MapMemory);
    HOOKD(UnmapMemory);
    HOOKD(FreeMemory);
    HOOKD(DestroyBuffer);
    HOOKD(BeginCommandBuffer);
    HOOKD(CmdBindVertexBuffers);
    HOOKD(CmdBindIndexBuffer);
    HOOKD(CreateGraphicsPipelines);
    HOOKD(CreateShaderModule);
    HOOKD(DestroyPipeline);
    HOOKD(CmdBindPipeline);
    HOOKD(CreateImage);
    HOOKD(CreateRenderPass);
    HOOKD(CreateRenderPass2);
    HOOKD(CmdBeginRenderPass);
    HOOKD(CmdBeginRenderPass2);
    HOOKD(GetDeviceQueue);
    HOOKD(GetDeviceQueue2);
    HOOKD(CreateImageView);
    HOOKD(CreateFramebuffer);
    HOOKD(CmdEndRenderPass);
    HOOKD(UpdateDescriptorSets);
    HOOKD(CmdBindDescriptorSets);
    HOOKD(DestroyImage);
#undef HOOKD
    return d->gdpa(device, name);
}

extern "C" __declspec(dllexport) VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL SkyRT_GetInstanceProcAddr(VkInstance instance,
                                                                                                     const char* name) {
#define HOOKI(fn) if (!strcmp(name, "vk" #fn)) return (PFN_vkVoidFunction)&SkyRT_##fn
    HOOKI(GetInstanceProcAddr);
    HOOKI(GetDeviceProcAddr);
    HOOKI(CreateInstance);
    HOOKI(DestroyInstance);
    HOOKI(CreateDevice);
#undef HOOKI
    InstanceData* d = instance ? GetInst(Key(instance)) : nullptr;
    if (!d) return nullptr;
    return d->gipa(instance, name);
}

// The SDK's vk_layer.h already declares vkNegotiateLoaderLayerInterfaceVersion with its own linkage,
// so we define a differently named function and export it under the real name via the linker.
#pragma comment(linker, "/EXPORT:vkNegotiateLoaderLayerInterfaceVersion=SkyRT_NegotiateLoaderLayerInterfaceVersion")
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
SkyRT_NegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* pVersionStruct) {
    if (!pVersionStruct || pVersionStruct->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT) return VK_ERROR_INITIALIZATION_FAILED;
    if (pVersionStruct->loaderLayerInterfaceVersion > 2) pVersionStruct->loaderLayerInterfaceVersion = 2;
    if (pVersionStruct->loaderLayerInterfaceVersion >= 2) {
        pVersionStruct->pfnGetInstanceProcAddr = SkyRT_GetInstanceProcAddr;
        pVersionStruct->pfnGetDeviceProcAddr = SkyRT_GetDeviceProcAddr;
        pVersionStruct->pfnGetPhysicalDeviceProcAddr = nullptr;
    }
    return VK_SUCCESS;
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID reserved) {
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(hModule);
    } else if (reason == DLL_PROCESS_DETACH && g_ready) {
        g_noLock = true;  // other threads are already dead, the mutex may be stuck
        Logf("DLL_PROCESS_DETACH (%s) - clean shutdown", reserved ? "process exit" : "FreeLibrary");
        FlushToDisk();
    }
    return TRUE;
}
