// SkyRT_Setup: one file that installs, starts and removes SkyRT, guided by Лучик.
// --demo          show everything, change nothing (also when the exe name contains "demo")
// --play          start the game with rays, no window (the desktop shortcut)
// --root <dir>    use this folder and a test registry key instead of the real ones
// --selftest      install, check, uninstall, write the result to stdout-less <root>\selftest.txt
// --shot <png> <screen>   render one screen to a file and quit
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#include <shlobj.h>
#include <shobjidl.h>
#include <commdlg.h>
#include <tlhelp32.h>
#include <d3d11.h>
#include <mmsystem.h>
#include <vulkan/vulkan.h>

#include <atomic>
#include <cmath>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "imgui.h"
#include "imgui_impl_dx11.h"
#include "imgui_impl_win32.h"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);

// ============================================================ text helpers
static std::wstring W(const std::string& s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring o(n > 0 ? n - 1 : 0, L'\0');
    if (n > 1) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &o[0], n);
    return o;
}
static std::string U(const std::wstring& s) {
    int n = WideCharToMultiByte(CP_UTF8, 0, s.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string o(n > 0 ? n - 1 : 0, '\0');
    if (n > 1) WideCharToMultiByte(CP_UTF8, 0, s.c_str(), -1, &o[0], n, nullptr, nullptr);
    return o;
}
static std::string LastError() {
    wchar_t* b = nullptr;
    const DWORD e = GetLastError();
    FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, e, 0, (wchar_t*)&b, 0, nullptr);
    std::string s = b ? U(b) : "код " + std::to_string(e);
    if (b) LocalFree(b);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    return s;
}

// ============================================================ paths and registry
static std::wstring g_testRoot;   // --root: nothing real is touched
static bool g_demo = false;       // --demo: no file, registry or game is touched, the steps are only played out
static bool g_demoInstalled = false;
static bool g_autoPrank = false;  // --autoprank: demo starts the install and presses "do not press" by itself
static std::wstring g_forceWay;   // --prankway up|left|none (with --autoprank): test the other monitor layouts
static const wchar_t* kLayerKey = L"SOFTWARE\\Khronos\\Vulkan\\ImplicitLayers";
static const wchar_t* kTestKey = L"SOFTWARE\\SkyRT_Test\\ImplicitLayers";
static const wchar_t* kManifest = L"VK_LAYER_SKYRT.json";
static const char* kSteamLine = "cmd /c \"set SKYRT_ENABLE=1&& %command%\"";

static const wchar_t* LayerKey() { return g_testRoot.empty() ? kLayerKey : kTestKey; }
static std::wstring InstallDir() {
    if (!g_testRoot.empty()) return g_testRoot;
    wchar_t b[MAX_PATH * 2] = {};
    GetEnvironmentVariableW(L"LOCALAPPDATA", b, MAX_PATH * 2);
    return std::wstring(b) + L"\\SkyRT";
}
static std::wstring ManifestPath() { return InstallDir() + L"\\" + kManifest; }
static std::wstring ShortcutPath() {
    if (!g_testRoot.empty()) return g_testRoot + L"\\Sky RTX.lnk";
    PWSTR p = nullptr;
    std::wstring o;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_Desktop, 0, nullptr, &p))) o = std::wstring(p) + L"\\Sky RTX.lnk";
    CoTaskMemFree(p);
    return o;
}
static bool FileThere(const std::wstring& p) {
    const DWORD a = GetFileAttributesW(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

// every registered copy of this layer
static std::vector<std::wstring> RegisteredManifests() {
    std::vector<std::wstring> o;
    HKEY k;
    if (RegOpenKeyExW(HKEY_CURRENT_USER, LayerKey(), 0, KEY_READ, &k) != ERROR_SUCCESS) return o;
    for (DWORD i = 0;; ++i) {
        wchar_t name[1024];
        DWORD n = 1024;
        if (RegEnumValueW(k, i, name, &n, nullptr, nullptr, nullptr, nullptr) != ERROR_SUCCESS) break;
        std::wstring s = name;
        const std::wstring tail = std::wstring(L"\\") + kManifest;
        if (s.size() > tail.size() && _wcsicmp(s.c_str() + s.size() - tail.size(), tail.c_str()) == 0) o.push_back(s);
    }
    RegCloseKey(k);
    return o;
}
static std::vector<std::wstring> OldCopies() {
    std::vector<std::wstring> o;
    for (const std::wstring& m : RegisteredManifests())
        if (_wcsicmp(m.c_str(), ManifestPath().c_str()) != 0) o.push_back(m);
    return o;
}
static bool IsInstalled() {
    if (g_demo) return g_demoInstalled;
    if (!FileThere(InstallDir() + L"\\SkyRT.dll")) return false;
    for (const std::wstring& m : RegisteredManifests())
        if (_wcsicmp(m.c_str(), ManifestPath().c_str()) == 0) return true;
    return false;
}
static bool GameRunning() {
    HANDLE s = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (s == INVALID_HANDLE_VALUE) return false;
    PROCESSENTRY32W e{sizeof e};
    bool found = false;
    for (BOOL ok = Process32FirstW(s, &e); ok; ok = Process32NextW(s, &e))
        if (_wcsicmp(e.szExeFile, L"Sky.exe") == 0) found = true;
    CloseHandle(s);
    return found;
}

// ============================================================ the game
static std::string ReadText(const std::wstring& p) {
    std::ifstream f(p.c_str(), std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}
static std::wstring FindGameInSteam() {
    wchar_t sp[MAX_PATH] = {};
    DWORD n = sizeof sp;
    if (RegGetValueW(HKEY_CURRENT_USER, L"Software\\Valve\\Steam", L"SteamPath", RRF_RT_REG_SZ, nullptr, sp, &n) != ERROR_SUCCESS) return L"";
    std::vector<std::wstring> libs{sp};
    const std::string vdf = ReadText(std::wstring(sp) + L"\\steamapps\\libraryfolders.vdf");
    for (size_t p = vdf.find("\"path\""); p != std::string::npos; p = vdf.find("\"path\"", p + 6)) {
        const size_t a = vdf.find('"', p + 6);
        if (a == std::string::npos) break;
        const size_t b = vdf.find('"', a + 1);
        if (b == std::string::npos) break;
        std::string v;
        for (size_t i = a + 1; i < b; ++i) {
            if (vdf[i] == '\\' && i + 1 < b && vdf[i + 1] == '\\') ++i;
            v += vdf[i];
        }
        libs.push_back(W(v));
    }
    for (const std::wstring& l : libs) {
        const std::wstring e = l + L"\\steamapps\\common\\Sky Children of the Light\\Sky.exe";
        if (FileThere(e)) return e;
    }
    return L"";
}
static std::wstring GamePathFile() { return InstallDir() + L"\\game_path.txt"; }
static std::wstring LoadGamePath() {
    const std::wstring s = W(ReadText(GamePathFile()));
    if (FileThere(s)) return s;
    return FindGameInSteam();
}
static void SaveGamePath(const std::wstring& p) {
    if (g_demo) return;
    std::ofstream f(GamePathFile().c_str(), std::ios::binary);
    f << U(p);
}
static bool StartGame(const std::wstring& exe, std::string& err) {
    if (g_demo) { err = "Это режим показа, игру я не запускаю. Но кнопка работает!"; return false; }
    if (!FileThere(exe)) { err = "Не нашёл Sky.exe по этому пути."; return false; }
    SetEnvironmentVariableW(L"SKYRT_ENABLE", L"1");
    std::wstring cmd = L"\"" + exe + L"\"";
    const std::wstring dir = exe.substr(0, exe.find_last_of(L'\\'));
    STARTUPINFOW si{sizeof si};
    PROCESS_INFORMATION pi{};
    const BOOL ok = CreateProcessW(exe.c_str(), &cmd[0], nullptr, nullptr, FALSE, 0, nullptr, dir.c_str(), &si, &pi);
    if (!ok) { err = "Игра не запустилась: " + LastError(); return false; }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return true;
}

// ============================================================ can this GPU trace rays?
struct GpuInfo { bool done = false; bool vulkan = false; bool rays = false; std::string name; };
static GpuInfo CheckGpu() {
    GpuInfo g;
    HMODULE vk = LoadLibraryW(L"vulkan-1.dll");
    if (!vk) { g.done = true; return g; }
    auto gipa = (PFN_vkGetInstanceProcAddr)GetProcAddress(vk, "vkGetInstanceProcAddr");
    auto create = gipa ? (PFN_vkCreateInstance)gipa(nullptr, "vkCreateInstance") : nullptr;
    VkInstance inst = VK_NULL_HANDLE;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    if (create && create(&ci, nullptr, &inst) == VK_SUCCESS) {
        g.vulkan = true;
        auto enumDev = (PFN_vkEnumeratePhysicalDevices)gipa(inst, "vkEnumeratePhysicalDevices");
        auto props = (PFN_vkGetPhysicalDeviceProperties)gipa(inst, "vkGetPhysicalDeviceProperties");
        auto enumExt = (PFN_vkEnumerateDeviceExtensionProperties)gipa(inst, "vkEnumerateDeviceExtensionProperties");
        auto destroy = (PFN_vkDestroyInstance)gipa(inst, "vkDestroyInstance");
        uint32_t n = 0;
        enumDev(inst, &n, nullptr);
        std::vector<VkPhysicalDevice> devs(n);
        if (n) enumDev(inst, &n, devs.data());
        for (VkPhysicalDevice d : devs) {
            VkPhysicalDeviceProperties p{};
            props(d, &p);
            uint32_t en = 0;
            enumExt(d, nullptr, &en, nullptr);
            std::vector<VkExtensionProperties> ex(en);
            if (en) enumExt(d, nullptr, &en, ex.data());
            bool rq = false, as = false;
            for (const VkExtensionProperties& e : ex) {
                if (!strcmp(e.extensionName, "VK_KHR_ray_query")) rq = true;
                if (!strcmp(e.extensionName, "VK_KHR_acceleration_structure")) as = true;
            }
            if (g.name.empty() || (rq && as && !g.rays)) { g.name = p.deviceName; g.rays = rq && as; }
        }
        destroy(inst, nullptr);
    }
    g.done = true;
    return g;
}

// ============================================================ install / uninstall
static bool WriteResource(int id, const std::wstring& path, std::string& err) {
    HRSRC r = FindResourceW(nullptr, MAKEINTRESOURCEW(id), MAKEINTRESOURCEW(10));
    HGLOBAL h = r ? LoadResource(nullptr, r) : nullptr;
    const void* p = h ? LockResource(h) : nullptr;
    if (!p) { err = "В установщике нет нужного файла: сборка повреждена."; return false; }
    HANDLE f = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) { err = "Не смог записать " + U(path) + ": " + LastError(); return false; }
    DWORD w = 0;
    const DWORD n = SizeofResource(nullptr, r);
    const BOOL ok = WriteFile(f, p, n, &w, nullptr);
    CloseHandle(f);
    if (!ok || w != n) { err = "Запись оборвалась: " + U(path); return false; }
    return true;
}
static bool MakeShortcut(const std::wstring& target, std::string& err) {
    IShellLinkW* l = nullptr;
    if (FAILED(CoCreateInstance(CLSID_ShellLink, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&l)))) { err = "Windows не дал создать ярлык."; return false; }
    l->SetPath(target.c_str());
    l->SetArguments(L"--play");
    l->SetIconLocation(target.c_str(), 0);
    l->SetDescription(L"Sky с лучами");
    IPersistFile* f = nullptr;
    HRESULT hr = l->QueryInterface(IID_PPV_ARGS(&f));
    if (SUCCEEDED(hr)) { hr = f->Save(ShortcutPath().c_str(), TRUE); f->Release(); }
    l->Release();
    if (FAILED(hr)) { err = "Не смог сохранить ярлык."; return false; }
    return true;
}

struct Job {
    std::atomic<bool> running{false}, finished{false}, ok{false};
    std::atomic<float> progress{0.0f};
    std::mutex m;
    std::string step, err;
    void Step(const char* s, float p) {
        { std::lock_guard<std::mutex> l(m); step = s; progress = p; }
        if (!g_demo && p < 1.0f) Sleep(700);   // the real steps take milliseconds: give people time to read them (and to find the red button)
    }
};
static Job g_job;

static bool DoInstall(Job& j, std::string& err) {
    if (g_demo) {
        const char* steps[] = {"Открываю папку для лучей", "Кладу слой на место", "Переношу себя, чтобы ярлык всегда работал", "Записываю слой в реестр", "Убираю старые записи слоя", "Делаю ярлык на рабочем столе"};
        for (int i = 0; i < 6; ++i) { j.Step(steps[i], (i + 1) / 7.0f); Sleep(1100); }
        j.Step("Готово", 1.0f);
        g_demoInstalled = true;
        return true;
    }
    if (GameRunning()) { err = "Игра запущена. Закрой Sky и попробуй ещё раз."; return false; }
    j.Step("Открываю папку для лучей", 0.1f);
    const std::wstring dir = InstallDir();
    const int mk = SHCreateDirectoryExW(nullptr, dir.c_str(), nullptr);
    if (mk != ERROR_SUCCESS && mk != ERROR_ALREADY_EXISTS && mk != ERROR_FILE_EXISTS) { err = "Не смог создать папку " + U(dir); return false; }
    j.Step("Кладу слой на место", 0.3f);
    if (!WriteResource(101, dir + L"\\SkyRT.dll", err)) return false;
    if (!WriteResource(102, ManifestPath(), err)) return false;
    j.Step("Переношу себя, чтобы ярлык всегда работал", 0.45f);
    wchar_t self[MAX_PATH * 2] = {};
    GetModuleFileNameW(nullptr, self, MAX_PATH * 2);
    const std::wstring me = dir + L"\\SkyRT_Setup.exe";
    if (_wcsicmp(self, me.c_str()) != 0 && !CopyFileW(self, me.c_str(), FALSE)) { err = "Не смог скопировать установщик: " + LastError(); return false; }
    j.Step("Записываю слой в реестр", 0.6f);
    HKEY k;
    if (RegCreateKeyExW(HKEY_CURRENT_USER, LayerKey(), 0, nullptr, 0, KEY_SET_VALUE, nullptr, &k, nullptr) != ERROR_SUCCESS) { err = "Windows не дал записать слой в реестр."; return false; }
    const DWORD zero = 0;
    const std::wstring man = ManifestPath();
    LSTATUS rs = RegSetValueExW(k, man.c_str(), 0, REG_DWORD, (const BYTE*)&zero, sizeof zero);
    if (rs == ERROR_SUCCESS) {
        j.Step("Убираю старые записи слоя", 0.75f);
        for (const std::wstring& o : OldCopies()) RegDeleteValueW(k, o.c_str());
    }
    RegCloseKey(k);
    if (rs != ERROR_SUCCESS) { err = "Windows не дал записать слой в реестр."; return false; }
    j.Step("Делаю ярлык на рабочем столе", 0.9f);
    if (!MakeShortcut(me, err)) return false;
    j.Step("Готово", 1.0f);
    return true;
}
static bool DoUninstall(Job& j, std::string& err) {
    if (g_demo) {
        j.Step("Стираю запись слоя", 0.3f); Sleep(1000);
        j.Step("Убираю файлы", 0.7f); Sleep(1000);
        j.Step("Готово", 1.0f);
        g_demoInstalled = false;
        return true;
    }
    if (GameRunning()) { err = "Игра запущена. Закрой Sky и попробуй ещё раз."; return false; }
    j.Step("Стираю запись слоя", 0.3f);
    HKEY k;
    if (RegOpenKeyExW(HKEY_CURRENT_USER, LayerKey(), 0, KEY_SET_VALUE, &k) == ERROR_SUCCESS) {
        RegDeleteValueW(k, ManifestPath().c_str());
        RegCloseKey(k);
    }
    j.Step("Убираю файлы", 0.7f);
    const std::wstring dir = InstallDir();
    DeleteFileW(ShortcutPath().c_str());
    DeleteFileW(ManifestPath().c_str());
    DeleteFileW(GamePathFile().c_str());
    if (!DeleteFileW((dir + L"\\SkyRT.dll").c_str()) && GetLastError() != ERROR_FILE_NOT_FOUND) { err = "Не смог удалить SkyRT.dll: " + LastError(); return false; }
    j.Step("Готово", 1.0f);
    return true;
}
// is the installed layer exactly the one inside this exe? an old copy must not pass for an install
static bool IsCurrent() {
    if (g_demo) return g_demoInstalled;
    if (!IsInstalled() || !FileThere(InstallDir() + L"\\SkyRT_Setup.exe")) return false;
    HRSRC r = FindResourceW(nullptr, MAKEINTRESOURCEW(101), MAKEINTRESOURCEW(10));
    HGLOBAL h = r ? LoadResource(nullptr, r) : nullptr;
    const void* p = h ? LockResource(h) : nullptr;
    if (!p) return false;
    const std::string have = ReadText(InstallDir() + L"\\SkyRT.dll");
    return have.size() == SizeofResource(nullptr, r) && memcmp(have.data(), p, have.size()) == 0;
}
template <class F> static void RunJob(F f) {
    g_job.finished = false; g_job.ok = false; g_job.running = true; g_job.progress = 0.0f;
    std::thread([f] {
        CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
        std::string e;
        const bool ok = f(g_job, e);
        { std::lock_guard<std::mutex> l(g_job.m); g_job.err = e; }
        g_job.ok = ok; g_job.running = false; g_job.finished = true;
        CoUninitialize();
    }).detach();
}

// ============================================================ D3D11
static ID3D11Device* g_dev;
static ID3D11DeviceContext* g_ctx;
static IDXGISwapChain* g_swap;
static ID3D11RenderTargetView* g_rtv;

static void MakeRtv() {
    ID3D11Texture2D* b = nullptr;
    g_swap->GetBuffer(0, IID_PPV_ARGS(&b));
    g_dev->CreateRenderTargetView(b, nullptr, &g_rtv);
    b->Release();
}
static LRESULT CALLBACK WndProc(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (ImGui_ImplWin32_WndProcHandler(h, m, w, l)) return 1;
    if (m == WM_SIZE && g_dev && w != SIZE_MINIMIZED) {
        if (g_rtv) { g_rtv->Release(); g_rtv = nullptr; }
        g_swap->ResizeBuffers(0, LOWORD(l), HIWORD(l), DXGI_FORMAT_UNKNOWN, 0);
        MakeRtv();
        return 0;
    }
    if (m == WM_DESTROY) { PostQuitMessage(0); return 0; }
    return DefWindowProcW(h, m, w, l);
}
static ID3D11ShaderResourceView* LoadSprite(int id) {
    HRSRC r = FindResourceW(nullptr, MAKEINTRESOURCEW(id), MAKEINTRESOURCEW(10));
    HGLOBAL hg = r ? LoadResource(nullptr, r) : nullptr;
    if (!hg) return nullptr;
    int w, h, c;
    stbi_uc* px = stbi_load_from_memory((const stbi_uc*)LockResource(hg), (int)SizeofResource(nullptr, r), &w, &h, &c, 4);
    if (!px) return nullptr;
    D3D11_TEXTURE2D_DESC d{};
    d.Width = w; d.Height = h; d.MipLevels = 1; d.ArraySize = 1;
    d.Format = DXGI_FORMAT_R8G8B8A8_UNORM; d.SampleDesc.Count = 1;
    d.Usage = D3D11_USAGE_IMMUTABLE; d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA s{px, (UINT)w * 4, 0};
    ID3D11Texture2D* t = nullptr;
    ID3D11ShaderResourceView* v = nullptr;
    if (SUCCEEDED(g_dev->CreateTexture2D(&d, &s, &t))) { g_dev->CreateShaderResourceView(t, nullptr, &v); t->Release(); }
    stbi_image_free(px);
    return v;
}

// ============================================================ the conversation
enum Pose { Hello, Think, Shock, Cover, Laugh, Dust, Serious, Guilty, Run, Joy, PoseCount };
enum Screen { ScrHello, ScrInstalling, ScrDone, ScrManage, ScrRemoved, ScrError };
static Screen g_screen = ScrHello;
static Pose g_pose = Hello;
static std::string g_say;
static double g_sayStart = 0;
static GpuInfo g_gpu;
static std::atomic<bool> g_gpuReady{false};
static std::wstring g_gamePath;
static std::string g_note;
static bool g_removing = false;
static int g_pokes = 0;
static int g_demoPose = -1;   // demo: the pose picked by hand

static void Say(Pose p, const std::string& s) {
    g_pose = p;
    if (s != g_say) { g_say = s; g_sayStart = ImGui::GetTime(); }
}
static void Go(Screen s) { g_screen = s; g_note.clear(); }

static bool BigButton(const char* t, float w = 0.0f) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.12f, 0.08f, 0.02f, 1.0f));
    const bool hit = ImGui::Button(t, ImVec2(w > 0 ? w : ImGui::GetContentRegionAvail().x, 44));
    ImGui::PopStyleColor();
    return hit;
}
static void CopyText(const char* s) {
    if (!OpenClipboard(nullptr)) return;
    EmptyClipboard();
    const std::wstring w = W(s);
    HGLOBAL h = GlobalAlloc(GMEM_MOVEABLE, (w.size() + 1) * sizeof(wchar_t));
    if (h) {
        memcpy(GlobalLock(h), w.c_str(), (w.size() + 1) * sizeof(wchar_t));
        GlobalUnlock(h);
        SetClipboardData(CF_UNICODETEXT, h);
    }
    CloseClipboard();
}
static void BrowseGame(HWND hwnd) {
    wchar_t f[MAX_PATH * 2] = L"Sky.exe";
    OPENFILENAMEW o{sizeof o};
    o.hwndOwner = hwnd;
    o.lpstrFilter = L"Sky.exe\0Sky.exe\0Все файлы\0*.*\0";
    o.lpstrFile = f;
    o.nMaxFile = MAX_PATH * 2;
    o.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetOpenFileNameW(&o)) { g_gamePath = f; SaveGamePath(g_gamePath); }
}

static void DrawBubble(ImVec2 pos, ImVec2 size) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImU32 fill = IM_COL32(38, 46, 74, 255), edge = IM_COL32(122, 142, 200, 255);
    dl->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), fill, 16.0f);
    dl->AddRect(pos, ImVec2(pos.x + size.x, pos.y + size.y), edge, 16.0f, 0, 2.0f);
    const float y = pos.y + size.y * 0.62f;
    dl->AddTriangleFilled(ImVec2(pos.x + 2, y - 12), ImVec2(pos.x - 20, y + 2), ImVec2(pos.x + 2, y + 14), fill);
    dl->AddLine(ImVec2(pos.x + 1, y - 12), ImVec2(pos.x - 20, y + 2), edge, 2.0f);
    dl->AddLine(ImVec2(pos.x - 20, y + 2), ImVec2(pos.x + 1, y + 14), edge, 2.0f);
    // the words appear one letter at a time
    const int want = (int)((ImGui::GetTime() - g_sayStart) * 42.0);
    const char* b = g_say.c_str();
    const char* e = b + g_say.size();
    const char* p = b;
    for (int n = 0; n < want && p < e; ++n) {
        ++p;
        while (p < e && ((unsigned char)*p & 0xC0) == 0x80) ++p;
    }
    ImGui::SetCursorScreenPos(ImVec2(pos.x + 20, pos.y + 16));
    ImGui::PushTextWrapPos(pos.x + size.x - 20 - ImGui::GetWindowPos().x);
    ImGui::TextUnformatted(b, p);
    ImGui::PopTextWrapPos();
}

// ============================================================ the "do not press" button: the progress bar breaks out of the window
// The bar stays where it is, shakes, a shot rings out, and its continuation stretches out of the window: it breaks the window edge
// (cracks inside, glass outside) and runs on to another monitor: right = 666%, left = -666% (it went "back"), above = straight up,
// one monitor = to the screen edge. It stays 5 seconds while Лучик peeks out from behind that monitor's edge, then it pulls back in.
// Click-through topmost layered windows carry the stretched part, the glass and Лучик; the real install keeps running.
enum PrankWay { WayRight, WayLeft, WayUp, WayNone };
struct Prank {
    bool on = false;
    double t0 = 0;
    PrankWay way = WayNone;
    RECT bar{};                 // the bar in the window, client coordinates
    POINT clientOrg{};          // screen position of the client area's corner
    POINT crack{}, crackClient{};
    LONG from = 0, to = 0;      // the stretched part runs from the window edge to here (x for left/right, y for up)
    LONG col = 0;               // for up: x of the column (screen)
    RECT tmon{};                // the monitor it stretches to (the home monitor when there is none)
    int percent = 0, finalPercent = -666;
    bool banged = false;
    HWND ext = nullptr, shards = nullptr, peek = nullptr;
};
static Prank g_prank;
static const int kBarW = 510, kBarH = 30;
static const double kShake = 0.6, kGrow = 1.4, kHoldFor = 5.0, kBackFor = 0.8;
static double PrankBreak() { return kShake; }
static double PrankArrive() { return kShake + kGrow; }
static double PrankReturn() { return PrankArrive() + kHoldFor; }
static double PrankHome() { return PrankReturn() + kBackFor; }
static double PrankEnd() { return PrankHome() + 5.2; }
static double PrankT() { return ImGui::GetTime() - g_prank.t0; }
static bool PrankLuchikAway() { return g_prank.on && PrankT() >= PrankBreak() && PrankT() < PrankHome(); }
static float PrankShakeX() { return g_prank.on && PrankT() < kShake ? (float)sin(PrankT() * 70.0) * 3.0f : 0.0f; }
// how far out the stretched part is, 0..1
static double PrankReach(double t) {
    if (t < kShake) return 0.0;
    if (t < PrankArrive()) { const double k = (t - kShake) / kGrow; return 1.0 - (1.0 - k) * (1.0 - k); }
    if (t < PrankReturn()) return 1.0;
    if (t < PrankHome()) { const double k = (t - PrankReturn()) / kBackFor; return 1.0 - k * k; }
    return 0.0;
}

static LRESULT CALLBACK ExtProc(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_PAINT) {
        PAINTSTRUCT ps;
        HDC dc = BeginPaint(h, &ps);
        RECT rc;
        GetClientRect(h, &rc);
        HBRUSH key = CreateSolidBrush(RGB(255, 0, 255));
        FillRect(dc, &rc, key);
        DeleteObject(key);
        HBRUSH g = CreateSolidBrush(RGB(170, 235, 70));   // the light green of the drawing
        HPEN pen = CreatePen(PS_SOLID, 2, RGB(60, 110, 20));
        HGDIOBJ ob = SelectObject(dc, g), op = SelectObject(dc, pen);
        Rectangle(dc, 0, 0, rc.right, rc.bottom);
        SelectObject(dc, ob); SelectObject(dc, op);
        DeleteObject(g); DeleteObject(pen);
        wchar_t t[32];
        swprintf(t, 32, L"%d%%", g_prank.percent);
        SetBkMode(dc, TRANSPARENT);
        SetTextColor(dc, RGB(30, 60, 10));
        const bool up = g_prank.way == WayUp;
        HFONT f = CreateFontW(22, 0, up ? 900 : 0, up ? 900 : 0, FW_BOLD, 0, 0, 0, DEFAULT_CHARSET, 0, 0, ANTIALIASED_QUALITY, 0, L"Segoe UI");
        HGDIOBJ of = SelectObject(dc, f);
        if (up) {   // written upwards, at the far end
            SIZE sz{};
            GetTextExtentPoint32W(dc, t, (int)wcslen(t), &sz);
            if (rc.bottom > sz.cx + 16) TextOutW(dc, (rc.right - sz.cy) / 2, sz.cx + 10, t, (int)wcslen(t));
        } else {
            RECT tr = rc;
            InflateRect(&tr, -12, 0);
            DrawTextW(dc, t, -1, &tr, DT_VCENTER | DT_SINGLELINE | (g_prank.way == WayRight ? DT_RIGHT : DT_LEFT));
        }
        SelectObject(dc, of);
        DeleteObject(f);
        EndPaint(h, &ps);
        return 0;
    }
    return DefWindowProcW(h, m, w, l);
}
static HWND MakeOverlay(const wchar_t* cls, WNDPROC proc) {
    WNDCLASSEXW wc{sizeof wc};
    if (!GetClassInfoExW(GetModuleHandleW(nullptr), cls, &wc)) {
        wc = WNDCLASSEXW{sizeof wc};
        wc.lpfnWndProc = proc;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = cls;
        RegisterClassExW(&wc);
    }
    return CreateWindowExW(WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE, cls, L"", WS_POPUP, 0, 0, 1, 1,
                           nullptr, nullptr, GetModuleHandleW(nullptr), nullptr);
}
// shows a BGRA premultiplied picture in a per-pixel-alpha window at (x, y)
static void PushPixels(HWND h, int x, int y, int w, int hh, const uint32_t* px) {
    if (w <= 0 || hh <= 0) { ShowWindow(h, SW_HIDE); return; }
    HDC sdc = GetDC(nullptr), mdc = CreateCompatibleDC(sdc);
    BITMAPINFO bi{};
    bi.bmiHeader = {sizeof(BITMAPINFOHEADER), w, -hh, 1, 32, BI_RGB};
    void* bits = nullptr;
    HBITMAP bm = CreateDIBSection(mdc, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (bm && bits) {
        memcpy(bits, px, (size_t)w * hh * 4);
        HGDIOBJ old = SelectObject(mdc, bm);
        POINT dst{x, y}, src{0, 0};
        SIZE sz{w, hh};
        BLENDFUNCTION bf{AC_SRC_OVER, 0, 255, AC_SRC_ALPHA};
        UpdateLayeredWindow(h, sdc, &dst, &sz, mdc, &src, 0, &bf, ULW_ALPHA);
        SelectObject(mdc, old);
        ShowWindow(h, SW_SHOWNOACTIVATE);
        SetWindowPos(h, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
    }
    if (bm) DeleteObject(bm);
    DeleteDC(mdc);
    ReleaseDC(nullptr, sdc);
}

// a gunshot made on the spot: a click, a burst of noise and a low thump, all fading fast
static std::vector<char> g_shot;
static void MakeShotSound() {
    const int rate = 22050, n = rate * 6 / 10;
    std::vector<int16_t> s(n);
    uint32_t seed = 12345;
    float lp = 0;
    for (int i = 0; i < n; ++i) {
        const float t = (float)i / rate;
        seed = seed * 1664525u + 1013904223u;
        const float noise = ((seed >> 9) & 0xFFFF) / 32768.0f - 1.0f;
        lp += (noise - lp) * 0.35f;
        float v = lp * expf(-t * 16.0f) * 1.3f + sinf(6.2831853f * 55.0f * t) * expf(-t * 9.0f) * 0.9f;
        if (i < 40) v += (i % 2 ? 1.0f : -1.0f) * 0.9f;
        s[i] = (int16_t)(std::max(-1.0f, std::min(1.0f, v)) * 30000);
    }
    const uint32_t data = n * 2;
    g_shot.resize(44 + data);
    char* p = g_shot.data();
    auto put32 = [&](size_t o, uint32_t v) { memcpy(p + o, &v, 4); };
    auto put16 = [&](size_t o, uint16_t v) { memcpy(p + o, &v, 2); };
    memcpy(p, "RIFF", 4); put32(4, 36 + data); memcpy(p + 8, "WAVEfmt ", 8);
    put32(16, 16); put16(20, 1); put16(22, 1); put32(24, rate); put32(28, rate * 2); put16(32, 2); put16(34, 16);
    memcpy(p + 36, "data", 4); put32(40, data);
    memcpy(p + 44, s.data(), data);
}

// the picture of Лучик peeking (premultiplied BGRA, 256x256), loaded once from the shock sprite
static std::vector<uint32_t> g_peekPx;
static void LoadPeekPixels() {
    if (!g_peekPx.empty()) return;
    HRSRC r = FindResourceW(nullptr, MAKEINTRESOURCEW(203), MAKEINTRESOURCEW(10));
    HGLOBAL hg = r ? LoadResource(nullptr, r) : nullptr;
    if (!hg) return;
    int w, h, c;
    stbi_uc* px = stbi_load_from_memory((const stbi_uc*)LockResource(hg), (int)SizeofResource(nullptr, r), &w, &h, &c, 4);
    if (!px || w != 256 || h != 256) { if (px) stbi_image_free(px); return; }
    g_peekPx.resize(256 * 256);
    for (int i = 0; i < 256 * 256; ++i) {
        const uint32_t a = px[i * 4 + 3];
        g_peekPx[i] = (a << 24) | ((px[i * 4 + 0] * a / 255) << 16) | ((px[i * 4 + 1] * a / 255) << 8) | (px[i * 4 + 2] * a / 255);
    }
    stbi_image_free(px);
}

static BOOL CALLBACK AddMonitor(HMONITOR, HDC, LPRECT r, LPARAM p) {
    ((std::vector<RECT>*)p)->push_back(*r);
    return TRUE;
}
static void PrankStart(HWND main, POINT barTopLeft) {
    std::vector<RECT> mons;
    EnumDisplayMonitors(nullptr, nullptr, AddMonitor, (LPARAM)&mons);
    MONITORINFO mi{sizeof mi};
    GetMonitorInfoW(MonitorFromWindow(main, MONITOR_DEFAULTTONEAREST), &mi);
    const RECT home = mi.rcMonitor;
    if (!g_forceWay.empty()) {   // test only: pretend the monitors stand differently
        mons.clear();
        const LONG w = home.right - home.left, h = home.bottom - home.top;
        if (g_forceWay == L"up") mons.push_back(RECT{home.left, home.top - h, home.right, home.top});
        if (g_forceWay == L"left") mons.push_back(RECT{home.left - w, home.top, home.left, home.bottom});
    }
    const RECT* up = nullptr;
    const RECT* right = nullptr;
    const RECT* left = nullptr;
    for (const RECT& r : mons) {
        if (EqualRect(&r, &home)) continue;
        if (r.bottom <= home.top + 8) { if (!up) up = &r; }
        else if (r.left >= home.right - 8) { if (!right) right = &r; }
        else if (r.right <= home.left + 8) { if (!left) left = &r; }
    }
    RECT cr;
    GetClientRect(main, &cr);
    POINT tl{0, 0}, br{cr.right, cr.bottom};
    ClientToScreen(main, &tl);
    ClientToScreen(main, &br);
    Prank& p = g_prank;
    const HWND ext = p.ext, shards = p.shards, peek = p.peek;
    p = Prank{};
    p.ext = ext; p.shards = shards; p.peek = peek;
    p.clientOrg = tl;
    p.bar = {barTopLeft.x - tl.x, barTopLeft.y - tl.y, barTopLeft.x - tl.x + kBarW, barTopLeft.y - tl.y + kBarH};
    p.tmon = home;
    const LONG midY = barTopLeft.y + kBarH / 2;
    auto cx = [](const RECT& r) { return (r.left + r.right) / 2; };
    auto cy = [](const RECT& r) { return (r.top + r.bottom) / 2; };
    if (up) {
        p.way = WayUp; p.tmon = *up; p.finalPercent = 666;
        p.col = barTopLeft.x + kBarW / 3;
        p.from = tl.y; p.to = cy(*up);
        p.crack = {p.col, tl.y};
    } else if (right) {
        p.way = WayRight; p.tmon = *right; p.finalPercent = 666;
        p.from = br.x; p.to = cx(*right);
        p.crack = {br.x, midY};
    } else if (left) {
        p.way = WayLeft; p.tmon = *left; p.finalPercent = -666;
        p.from = tl.x; p.to = cx(*left);
        p.crack = {tl.x, midY};
    } else {
        p.way = WayNone; p.finalPercent = -666;
        p.from = tl.x; p.to = home.left;
        p.crack = {tl.x, midY};
    }
    p.crackClient = {p.crack.x - tl.x, p.crack.y - tl.y};
    if (!p.ext) p.ext = MakeOverlay(L"SkyRTStretchedBar", ExtProc);
    if (!p.shards) p.shards = MakeOverlay(L"SkyRTShards", DefWindowProcW);
    if (!p.peek) p.peek = MakeOverlay(L"SkyRTPeek", DefWindowProcW);
    SetLayeredWindowAttributes(p.ext, RGB(255, 0, 255), 0, LWA_COLORKEY);
    if (g_shot.empty()) MakeShotSound();
    LoadPeekPixels();
    p.percent = (int)(g_job.progress.load() * 100.0f);
    p.t0 = ImGui::GetTime();
    p.on = true;
}
static void PrankStop() {
    for (HWND h : {g_prank.ext, g_prank.shards, g_prank.peek}) if (h) ShowWindow(h, SW_HIDE);
    g_prank.on = false;
}
// the stretched part outside the window, screen coordinates (empty when it is all inside)
static RECT PrankExtRect(double reach) {
    const Prank& p = g_prank;
    const LONG len = (LONG)((p.to - p.from) * reach);
    if (p.way == WayUp) return RECT{p.col - kBarH / 2, p.from + len, p.col + kBarH / 2, p.from};
    const LONG y = p.clientOrg.y + p.bar.top;
    if (len >= 0) return RECT{p.from, y, p.from + len, y + kBarH};
    return RECT{p.from + len, y, p.from, y + kBarH};
}

static void PrankShards(double since) {
    const int S = 520;
    static std::vector<uint32_t> px(S * S);
    std::fill(px.begin(), px.end(), 0u);
    const float dx = g_prank.way == WayRight ? 1.0f : g_prank.way == WayUp ? 0.0f : -1.0f;
    const float dy = g_prank.way == WayUp ? -1.0f : 0.0f;
    uint32_t seed = 777;
    auto rnd = [&] { seed = seed * 1664525u + 1013904223u; return ((seed >> 8) & 0xFFFF) / 65535.0f; };
    const float fade = std::max(0.0f, 1.0f - (float)since / 1.3f);
    if (fade <= 0.0f) { ShowWindow(g_prank.shards, SW_HIDE); return; }
    for (int i = 0; i < 46; ++i) {
        const float sp = 260.0f + rnd() * 520.0f, side = (rnd() - 0.5f) * 1.6f;
        const float vx = (dx + (dy != 0 ? side : 0)) * sp, vy = (dy + (dx != 0 ? side : 0)) * sp;
        const float x = S / 2 + vx * (float)since, y = S / 2 + vy * (float)since + 900.0f * (float)(since * since) * 0.5f;
        const int sz = 4 + (int)(rnd() * 8);
        const uint32_t shade = rnd() < 0.5f ? 0xFFDCEBFFu : 0xFF8FA8D8u;   // glass: white-blue and blue
        const uint32_t a = (uint32_t)(255 * fade);
        const uint32_t c = (a << 24) | ((((shade >> 16) & 255) * a / 255) << 16) | ((((shade >> 8) & 255) * a / 255) << 8) | ((shade & 255) * a / 255);
        for (int yy = (int)y; yy < (int)y + sz; ++yy)
            for (int xx = (int)x; xx < (int)x + sz; ++xx)
                if (xx >= 0 && yy >= 0 && xx < S && yy < S) px[yy * S + xx] = c;
    }
    PushPixels(g_prank.shards, g_prank.crack.x - S / 2, g_prank.crack.y - S / 2, S, S, px.data());
}

// Лучик slides out from behind the edge of the monitor the bar went to; only the part inside that monitor is shown
static void PrankPeek(double t) {
    Prank& p = g_prank;
    const double in0 = PrankArrive() + 0.4, out1 = PrankReturn();
    if (t < in0 || t > out1 || g_peekPx.empty()) { if (IsWindowVisible(p.peek)) ShowWindow(p.peek, SW_HIDE); return; }
    double k = std::min(1.0, (t - in0) / 0.5);
    if (t > out1 - 0.6) k = std::min(k, (out1 - t) / 0.6);
    const int show = (int)(160 * (1.0 - (1.0 - k) * (1.0 - k)));   // how far he leans out
    const RECT& m = p.tmon;
    const LONG barY = p.clientOrg.y + p.bar.top;
    auto clampY = [&](LONG y) { return std::max<LONG>(m.top, std::min<LONG>(m.bottom - 256, y)); };
    int sx, sy;   // where the whole 256x256 picture would be
    if (p.way == WayRight) { sx = m.left - 256 + show; sy = clampY(barY + 60); }
    else if (p.way == WayLeft) { sx = m.right - show; sy = clampY(barY + 60); }
    else if (p.way == WayUp) { sx = p.col + 60; sy = m.bottom - show; }
    else { sx = m.left - 256 + show; sy = clampY(barY + 60); }
    const int x0 = std::max<int>(sx, m.left), y0 = std::max<int>(sy, m.top);
    const int x1 = std::min<int>(sx + 256, m.right), y1 = std::min<int>(sy + 256, m.bottom);
    const int w = x1 - x0, h = y1 - y0;
    if (w <= 0 || h <= 0) { ShowWindow(p.peek, SW_HIDE); return; }
    std::vector<uint32_t> crop((size_t)w * h);
    for (int y = 0; y < h; ++y) memcpy(&crop[(size_t)y * w], &g_peekPx[(size_t)(y0 - sy + y) * 256 + (x0 - sx)], (size_t)w * 4);
    PushPixels(p.peek, x0, y0, w, h, crop.data());
}

static bool PrankUpdate() {
    if (!g_prank.on) return false;
    const double t = PrankT();
    if (t >= PrankEnd()) { PrankStop(); return false; }
    if (!g_prank.banged && t >= PrankBreak()) {
        g_prank.banged = true;
        PlaySoundW((LPCWSTR)g_shot.data(), nullptr, SND_MEMORY | SND_ASYNC | SND_NODEFAULT);
    }
    if (g_prank.banged) PrankShards(t - PrankBreak());
    PrankPeek(t);
    const double reach = PrankReach(t);
    const int real = (int)(g_job.progress.load() * 100.0f);
    g_prank.percent = (int)(real + (g_prank.finalPercent - real) * reach);
    const RECT r = PrankExtRect(reach);
    if (r.right - r.left < 2 || r.bottom - r.top < 2) { if (IsWindowVisible(g_prank.ext)) ShowWindow(g_prank.ext, SW_HIDE); }
    else {
        SetWindowPos(g_prank.ext, HWND_TOPMOST, r.left, r.top, r.right - r.left, r.bottom - r.top, SWP_NOACTIVATE | SWP_SHOWWINDOW);
        InvalidateRect(g_prank.ext, nullptr, FALSE);
    }
    if (g_autoPrank) {   // test trace: positions only, never a picture of the user's screen
        wchar_t tmp[MAX_PATH];
        GetTempPathW(MAX_PATH, tmp);
        static FILE* tf = _wfopen((std::wstring(tmp) + L"skyrt_prank.txt").c_str(), L"w");
        RECT pr{};
        if (IsWindowVisible(g_prank.peek)) GetWindowRect(g_prank.peek, &pr);
        if (tf) { fprintf(tf, "%.2f ext %ld %ld %ld %ld %d peek %ld %ld %ld %ld shards %d\n", t, r.left, r.top, r.right, r.bottom, g_prank.percent, pr.left, pr.top, pr.right, pr.bottom, IsWindowVisible(g_prank.shards) ? 1 : 0); fflush(tf); }
    }
    return true;
}
// inside the window: the stretched part between the bar and the broken edge, the hole and the cracks (they heal after it is back)
static void PrankDrawInside(ImDrawList* dl) {
    if (!g_prank.on || !g_prank.banged) return;
    const double t = PrankT();
    const Prank& p = g_prank;
    if (PrankReach(t) > 0.0) {
        ImVec2 a, b;
        if (p.way == WayUp) { const float x = (float)(p.col - p.clientOrg.x); a = ImVec2(x - kBarH / 2, 0); b = ImVec2(x + kBarH / 2, (float)p.bar.top); }
        else if (p.way == WayRight) { a = ImVec2((float)p.bar.right, (float)p.bar.top); b = ImVec2((float)p.crackClient.x, (float)p.bar.bottom); }
        else { a = ImVec2((float)p.crackClient.x, (float)p.bar.top); b = ImVec2((float)p.bar.left, (float)p.bar.bottom); }
        dl->AddRectFilled(a, b, IM_COL32(170, 235, 70, 255));
        dl->AddRect(a, b, IM_COL32(60, 110, 20, 255), 0.0f, 0, 2.0f);
    }
    const float al = t < PrankHome() ? 1.0f : std::max(0.0f, 1.0f - (float)(t - PrankHome()) / 0.8f);
    if (al <= 0.0f) return;
    const ImVec2 c((float)p.crackClient.x, (float)p.crackClient.y);
    uint32_t seed = 4242;
    auto rnd = [&] { seed = seed * 1664525u + 1013904223u; return ((seed >> 8) & 0xFFFF) / 65535.0f; };
    ImVec2 hole[14];
    for (int i = 0; i < 14; ++i) {
        const float ang = i / 14.0f * 6.2831853f, r = 26.0f + rnd() * 22.0f;
        hole[i] = ImVec2(c.x + cosf(ang) * r, c.y + sinf(ang) * r);
    }
    dl->AddConvexPolyFilled(hole, 14, IM_COL32(4, 6, 12, (int)(170 * al)));
    for (int i = 0; i < 11; ++i) {
        float ang = rnd() * 6.2831853f, len = 50.0f + rnd() * 120.0f;
        ImVec2 q0 = c;
        for (int s = 0; s < 4; ++s) {
            ang += (rnd() - 0.5f) * 0.8f;
            const ImVec2 q(q0.x + cosf(ang) * len / 4, q0.y + sinf(ang) * len / 4);
            dl->AddLine(q0, q, IM_COL32(200, 220, 255, (int)(200 * al)), 2.0f - s * 0.35f);
            q0 = q;
        }
    }
}
static void PrankConversation() {
    const double t = PrankT();
    if (t < PrankBreak()) Say(Shock, "Ой… она дрожит. Не трогайте её, пожалуйста!");
    else if (t < PrankHome()) Say(Shock, "…");
    else if (t < PrankHome() + 2.0) Say(Laugh, "АХАХАХА! Видели бы вы своё лицо! Ой, не могу…");
    else if (t < PrankHome() + 3.2) Say(Dust, "Кхм. Извините. Это было непрофессионально.");
    else Say(Serious, "Установка продолжается в штатном режиме. Или нет…");
}

static void UpdateConversation() {
    if (g_prank.on) { PrankConversation(); return; }
    if (g_screen == ScrHello) {
        if (!g_gpuReady) { Say(Think, "Ой, здравствуйте… Секундочку, я посмотрю, что у вас за видеокарта."); return; }
        if (!g_gpu.vulkan) Say(Guilty, "Простите… Я не нашёл Vulkan. Лучи без него не заработают. Обновите, пожалуйста, драйвер видеокарты.");
        else if (!g_gpu.rays) Say(Guilty, "Ой… " + g_gpu.name + " не умеет аппаратные лучи. Поставить я могу, но они не включатся. Простите.");
        else if (IsInstalled() || !RegisteredManifests().empty()) Say(Hello, "Здравствуйте! У вас уже стоит старая версия лучей. Давайте я её обновлю? Настройки останутся. " + g_gpu.name + " подходит.");
        else Say(Hello, "Здравствуйте! Вы правда хотите поставить лучи в Sky? Я постараюсь, честно. " + g_gpu.name + " подходит, она у вас хорошая.");
    } else if (g_screen == ScrInstalling) {
        std::lock_guard<std::mutex> l(g_job.m);
        Say(Run, g_job.step + "…");
    } else if (g_screen == ScrDone) {
        Say(Joy, g_gamePath.empty() ? "Получилось! Только я не нашёл игру. Покажите мне, где лежит Sky.exe, пожалуйста." : "Получилось! Лучи на месте. Если что-то сломается, я не виноват. Наверное.");
    } else if (g_screen == ScrManage) {
        Say(Serious, "Лучи уже установлены. Чем могу помочь?");
    } else if (g_screen == ScrRemoved) {
        Say(Cover, "Всё убрал. Мне немножко грустно, но я понимаю.");
    }
}

static void DrawActions(HWND hwnd) {
    if (g_screen == ScrHello) {
        ImGui::BeginDisabled(!g_gpuReady.load());
        if (BigButton("Да, ставь")) { g_removing = false; Go(ScrInstalling); RunJob(DoInstall); }
        ImGui::EndDisabled();
        if (BigButton("Я случайно нажал, выпустите меня")) PostQuitMessage(0);
    } else if (g_screen == ScrInstalling) {
        const float shake = PrankShakeX();   // the bar stays in place; before the shot it shakes
        if (shake != 0.0f) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + shake);
        ImGui::ProgressBar(g_job.progress.load(), ImVec2(kBarW, kBarH), "");
        const ImVec2 barMin = ImVec2(ImGui::GetItemRectMin().x - shake, ImGui::GetItemRectMin().y);
        ImGui::Spacing();
        ImGui::BeginDisabled(g_prank.on);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.80f, 0.20f, 0.20f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.92f, 0.28f, 0.28f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 1, 1, 1));
        static bool autoDone = false;   // --autoprank (demo only): presses the button by itself, for testing the animation
        const bool autoNow = g_autoPrank && !autoDone && g_job.progress.load() > 0.1f;
        if (autoNow) autoDone = true;
        if (ImGui::Button("Не нажимать", ImVec2(170, 34)) || autoNow) {
            POINT p{(LONG)barMin.x, (LONG)barMin.y};
            ClientToScreen(hwnd, &p);
            PrankStart(hwnd, p);
        }
        ImGui::PopStyleColor(3);
        ImGui::EndDisabled();
        if (g_prank.on && (ImGui::IsKeyPressed(ImGuiKey_Escape) || ImGui::IsMouseClicked(ImGuiMouseButton_Right))) PrankStop();   // a way out for anyone in a hurry
        if (g_job.finished && !g_prank.on) {
            if (g_job.ok) {
                if (g_removing) Go(ScrRemoved);
                else { g_gamePath = LoadGamePath(); if (!g_gamePath.empty()) SaveGamePath(g_gamePath); Go(ScrDone); }
            } else {
                std::lock_guard<std::mutex> l(g_job.m);
                Say(Guilty, "Простите, не вышло. " + g_job.err);
                Go(ScrError);
            }
        }
    } else if (g_screen == ScrDone || g_screen == ScrManage) {
        ImGui::BeginDisabled(g_gamePath.empty());
        if (BigButton("Играть с лучами")) {
            std::string e;
            if (StartGame(g_gamePath, e)) PostQuitMessage(0); else { Say(Guilty, e); Go(ScrError); }
        }
        ImGui::EndDisabled();
        const float half = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
        if (BigButton("Для запуска из Steam", half)) { CopyText(kSteamLine); g_note = "Скопировал. Вставьте в Steam: Sky → Свойства → Параметры запуска."; }
        ImGui::SameLine();
        if (BigButton("Где Sky.exe?", half)) BrowseGame(hwnd);
        if (g_screen == ScrManage) {
            if (BigButton("Переустановить", half)) { g_removing = false; Go(ScrInstalling); RunJob(DoInstall); }
            ImGui::SameLine();
            if (BigButton("Удалить", half)) { g_removing = true; Go(ScrInstalling); RunJob(DoUninstall); }
        }
        if (!g_note.empty()) ImGui::TextWrapped("%s", g_note.c_str());
    } else {
        if (g_screen == ScrError && BigButton("Попробовать ещё раз")) { Go(IsCurrent() ? ScrManage : ScrHello); }
        if (BigButton("Закрыть")) PostQuitMessage(0);
    }
    if (g_demo && g_screen != ScrInstalling) {
        const float half = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
        ImGui::Spacing();
        if (BigButton("Показать другую позу", half)) g_demoPose = (g_demoPose + 1) % PoseCount;
        ImGui::SameLine();
        if (BigButton("Показать ошибку", half)) { g_demoPose = -1; Say(Guilty, "Простите, не вышло. Игра запущена. Закрой Sky и попробуй ещё раз."); Go(ScrError); }
    }
}

// ============================================================ entry
static std::wstring ArgAfter(int argc, wchar_t** argv, const wchar_t* key, int n = 1) {
    for (int i = 1; i + n < argc; ++i)
        if (!wcscmp(argv[i], key)) return argv[i + n];
    return L"";
}
static bool HasArg(int argc, wchar_t** argv, const wchar_t* key) {
    for (int i = 1; i < argc; ++i)
        if (!wcscmp(argv[i], key)) return true;
    return false;
}
static int SelfTest() {
    std::string out, e;
    Job j;
    const bool in = DoInstall(j, e);
    out += "install " + std::string(in ? "ok" : "FAILED " + e) + "\n";
    out += "installed " + std::to_string(IsInstalled()) + " current " + std::to_string(IsCurrent()) + "\n";
    out += "dll " + std::to_string(FileThere(InstallDir() + L"\\SkyRT.dll")) + " manifest " + std::to_string(FileThere(ManifestPath())) + " shortcut " + std::to_string(FileThere(ShortcutPath())) + "\n";
    const GpuInfo g = CheckGpu();
    out += "gpu vulkan " + std::to_string(g.vulkan) + " rays " + std::to_string(g.rays) + " " + g.name + "\n";
    out += "game " + U(FindGameInSteam()) + "\n";
    const bool un = DoUninstall(j, e);
    out += "uninstall " + std::string(un ? "ok" : "FAILED " + e) + "\n";
    out += "installed after " + std::to_string(IsInstalled()) + "\n";
    out += "dll after " + std::to_string(FileThere(InstallDir() + L"\\SkyRT.dll")) + "\n";
    std::ofstream f((InstallDir() + L"\\selftest.txt").c_str(), std::ios::binary);
    f << out;
    return in && un ? 0 : 1;
}
static void SavePng(const std::wstring& path) {
    ID3D11Texture2D* b = nullptr;
    g_swap->GetBuffer(0, IID_PPV_ARGS(&b));
    D3D11_TEXTURE2D_DESC d{};
    b->GetDesc(&d);
    d.Usage = D3D11_USAGE_STAGING; d.BindFlags = 0; d.CPUAccessFlags = D3D11_CPU_ACCESS_READ; d.MiscFlags = 0;
    ID3D11Texture2D* s = nullptr;
    g_dev->CreateTexture2D(&d, nullptr, &s);
    g_ctx->CopyResource(s, b);
    D3D11_MAPPED_SUBRESOURCE m{};
    if (SUCCEEDED(g_ctx->Map(s, 0, D3D11_MAP_READ, 0, &m))) {
        std::vector<unsigned char> px(d.Width * d.Height * 4);
        for (UINT y = 0; y < d.Height; ++y) memcpy(&px[y * d.Width * 4], (char*)m.pData + y * m.RowPitch, d.Width * 4);
        for (size_t i = 3; i < px.size(); i += 4) px[i] = 255;
        stbi_write_png(U(path).c_str(), d.Width, d.Height, 4, px.data(), d.Width * 4);
        g_ctx->Unmap(s, 0);
    }
    s->Release();
    b->Release();
}

int WINAPI wWinMain(HINSTANCE inst, HINSTANCE, PWSTR, int show) {
    int argc = 0;
    wchar_t** argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    g_testRoot = ArgAfter(argc, argv, L"--root");
    {
        wchar_t self[MAX_PATH * 2] = {};
        GetModuleFileNameW(nullptr, self, MAX_PATH * 2);
        const wchar_t* base = wcsrchr(self, L'\\');
        std::wstring name = base ? base + 1 : self;
        for (wchar_t& c : name) c = towlower(c);
        g_demo = HasArg(argc, argv, L"--demo") || name.find(L"demo") != std::wstring::npos;
    }
    if (HasArg(argc, argv, L"--selftest")) return SelfTest();
    bool startUi = true;
    if (HasArg(argc, argv, L"--play") && !g_demo) {
        std::string e;
        const std::wstring g = LoadGamePath();
        if (!g.empty() && IsInstalled() && StartGame(g, e)) startUi = false;
    }
    if (!startUi) return 0;
    const std::wstring shot = ArgAfter(argc, argv, L"--shot");
    const std::wstring shotScreen = ArgAfter(argc, argv, L"--shot", 2);

    WNDCLASSEXW wc{sizeof wc, CS_CLASSDC, WndProc, 0, 0, inst, nullptr, LoadCursor(nullptr, IDC_ARROW), nullptr, nullptr, L"SkyRTSetup", nullptr};
    wc.hIcon = LoadIconW(inst, MAKEINTRESOURCEW(1));
    wc.hIconSm = wc.hIcon;
    RegisterClassExW(&wc);
    const DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
    RECT rc{0, 0, 860, 520};
    AdjustWindowRect(&rc, style, FALSE);
    HWND hwnd = CreateWindowW(wc.lpszClassName, L"Установка лучей в Sky", style, 200, 120, rc.right - rc.left, rc.bottom - rc.top, nullptr, nullptr, inst, nullptr);
    DXGI_SWAP_CHAIN_DESC sd{};
    sd.BufferCount = 2; sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hwnd; sd.SampleDesc.Count = 1; sd.Windowed = TRUE; sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    if (FAILED(D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &sd, &g_swap, &g_dev, nullptr, &g_ctx)) &&
        FAILED(D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &sd, &g_swap, &g_dev, nullptr, &g_ctx))) {
        MessageBoxW(nullptr, L"Не удалось открыть окно: Direct3D 11 недоступен.", L"SkyRT", MB_ICONERROR);
        return 1;
    }
    MakeRtv();
    ShowWindow(hwnd, show);
    UpdateWindow(hwnd);

    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    ImFontConfig fc;
    if (FileThere(L"C:\\Windows\\Fonts\\segoeui.ttf")) io.Fonts->AddFontFromFileTTF("C:\\Windows\\Fonts\\segoeui.ttf", 20.0f, &fc, io.Fonts->GetGlyphRangesCyrillic());
    else if (FileThere(L"C:\\Windows\\Fonts\\arial.ttf")) io.Fonts->AddFontFromFileTTF("C:\\Windows\\Fonts\\arial.ttf", 20.0f, &fc, io.Fonts->GetGlyphRangesCyrillic());
    ImGui::StyleColorsDark();
    ImGuiStyle& st = ImGui::GetStyle();
    st.FrameRounding = 10.0f; st.WindowRounding = 0.0f; st.WindowBorderSize = 0.0f;
    st.Colors[ImGuiCol_WindowBg] = ImVec4(0.09f, 0.11f, 0.18f, 1.0f);
    st.Colors[ImGuiCol_Button] = ImVec4(0.91f, 0.68f, 0.26f, 1.0f);
    st.Colors[ImGuiCol_ButtonHovered] = ImVec4(1.0f, 0.79f, 0.36f, 1.0f);
    st.Colors[ImGuiCol_ButtonActive] = ImVec4(0.80f, 0.56f, 0.18f, 1.0f);
    st.Colors[ImGuiCol_PlotHistogram] = ImVec4(0.35f, 0.86f, 0.45f, 1.0f);
    st.Colors[ImGuiCol_FrameBg] = ImVec4(0.17f, 0.21f, 0.33f, 1.0f);
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_dev, g_ctx);

    ID3D11ShaderResourceView* spr[PoseCount] = {};
    for (int i = 0; i < PoseCount; ++i) spr[i] = LoadSprite(201 + i);

    std::thread([] { g_gpu = CheckGpu(); g_gpuReady = true; }).detach();
    g_gamePath = LoadGamePath();
    if (IsCurrent()) g_screen = ScrManage;
    g_autoPrank = g_demo && HasArg(argc, argv, L"--autoprank");
    if (g_autoPrank) g_forceWay = ArgAfter(argc, argv, L"--prankway");
    if (g_autoPrank) { g_screen = ScrInstalling; RunJob(DoInstall); }
    if (shotScreen == L"done") g_screen = ScrDone;
    if (shotScreen == L"manage") g_screen = ScrManage;
    if (shotScreen == L"removed") g_screen = ScrRemoved;
    if (shotScreen == L"hello") g_screen = ScrHello;

    int frames = 0;
    bool quit = false;
    while (!quit) {
        MSG m;
        while (PeekMessageW(&m, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&m);
            DispatchMessageW(&m);
            if (m.message == WM_QUIT) quit = true;
        }
        if (quit) break;
        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();
        const ImVec2 ds = io.DisplaySize;
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ds);
        ImGui::Begin("##root", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus);
        PrankUpdate();
        UpdateConversation();
        if (g_demo && g_demoPose >= 0) g_pose = (Pose)g_demoPose;
        ImDrawList* dl = ImGui::GetWindowDrawList();
        if (g_demo) dl->AddText(ImVec2(20, 10), IM_COL32(255, 205, 110, 255), "РЕЖИМ ПОКАЗА");
        const float t = (float)ImGui::GetTime();
        const float bob = sinf(t * 2.2f) * 5.0f;
        const ImVec2 sp(30, 170 + bob);
        const ImVec2 c(sp.x + 128, sp.y + 140);
        for (int i = 40; i > 0; --i)   // a soft glow behind him
            dl->AddCircleFilled(c, 60.0f + i * 3.5f, IM_COL32(255, 205, 110, 2), 64);
        if (spr[g_pose] && !PrankLuchikAway()) dl->AddImage((ImTextureID)spr[g_pose], sp, ImVec2(sp.x + 256, sp.y + 256));
        ImGui::SetCursorScreenPos(sp);
        ImGui::InvisibleButton("##luchik", ImVec2(256, 256));
        if (ImGui::IsItemClicked() && ++g_pokes >= 5) { g_pokes = 0; Say(Shock, "Ой! Ну щекотно же… Я тут вообще-то файлы раскладываю."); }
        DrawBubble(ImVec2(320, 28), ImVec2(ds.x - 350, 170));
        PrankDrawInside(dl);
        ImGui::SetCursorScreenPos(ImVec2(320, 222));
        ImGui::BeginChild("##actions", ImVec2(ds.x - 350, 280));
        DrawActions(hwnd);
        ImGui::EndChild();
        ImGui::End();
        ImGui::Render();
        const float clear[4] = {0.09f, 0.11f, 0.18f, 1.0f};
        g_ctx->OMSetRenderTargets(1, &g_rtv, nullptr);
        g_ctx->ClearRenderTargetView(g_rtv, clear);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        g_swap->Present(1, 0);
        if (!shot.empty() && ImGui::GetTime() > 3.5 && g_gpuReady) { SavePng(shot); break; }
        if (g_autoPrank && g_screen == ScrDone && !g_prank.on) break;
        ++frames;
    }
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    return 0;
}
