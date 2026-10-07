#!/usr/bin/env python3
"""
SkyRT Panel - installer + control panel for the SkyRT ray tracing layer (Sky: Children of the Light, PC).

  * Install / uninstall the Vulkan layer with one button (copies SkyRT.dll + manifest to %LOCALAPPDATA%\\SkyRT
    and registers it for the current user).
  * Start the game with or without rays (sets SKYRT_ENABLE=1 only for the game process).
  * Edit every SkyRT.cfg setting with sliders; the layer re-reads the file while the game runs (live).
  * Show the newest SkyRT_*.log.

Put this script (or the exe built from it) in the same folder as SkyRT.dll and VK_LAYER_SKYRT.json.
Requires: Python 3.8+, PyQt5  (pip install PyQt5)
"""
import os
import sys
import json
import glob
import shutil
import subprocess

from PyQt5.QtCore import Qt, QTimer, QUrl
from PyQt5.QtGui import QDesktopServices, QFont, QPalette, QColor
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                             QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSlider,
                             QTabWidget, QVBoxLayout, QWidget, QDoubleSpinBox)

try:
    import winreg  # Windows only
except ImportError:  # the panel still opens elsewhere (useful for testing), install is disabled
    winreg = None

APP_NAME = "SkyRT"
REG_PATH = r"SOFTWARE\Khronos\Vulkan\ImplicitLayers"
DLL = "SkyRT.dll"
MANIFEST = "VK_LAYER_SKYRT.json"

# ---------------------------------------------------------------- texts (ru / en)
TXT = {
    "ru": {
        "title": "SkyRT - лучи для Sky",
        "tab_game": "Игра", "tab_gfx": "Графика", "tab_log": "Лог",
        "st_dll": "Файлы мода", "st_layer": "Слой Vulkan", "st_game": "Игра",
        "ok_dll": "найдены", "no_dll": "НЕ НАЙДЕНЫ (положи SkyRT.dll и VK_LAYER_SKYRT.json рядом с панелью)",
        "layer_on": "установлен", "layer_off": "не установлен", "layer_na": "доступно только в Windows",
        "game_set": "указана", "game_unset": "не указана",
        "install": "Установить лучи", "uninstall": "Удалить лучи",
        "game_path": "Путь к Sky.exe", "browse": "Обзор...",
        "play_rt": "▶  Играть с лучами", "play_plain": "Играть без лучей",
        "installed_ok": "Слой установлен. Теперь запускай игру кнопкой «Играть с лучами».",
        "uninstalled_ok": "Слой удалён из реестра. Папку с файлами можно удалить вручную.",
        "err": "Ошибка", "no_game": "Сначала укажи путь к Sky.exe.", "no_layer": "Сначала установи лучи.",
        "preset": "Пресет", "p_low": "Низкий", "p_mid": "Средний", "p_high": "Высокий", "p_ultra": "Ультра", "p_custom": "Свой",
        "g_main": "Общее", "g_sh": "Тени", "g_ao": "Затенение в щелях (AO)", "g_gi": "Отражённый свет (GI)",
        "g_dn": "Шумоподавление", "g_obj": "Объекты", "g_tex": "Текстуры", "g_sun": "Солнце", "g_light": "Свет от огня и ламп",
        "rt_on": "Лучи включены", "shadows": "Тени", "ao": "AO", "gi": "GI",
        "strength": "Сила", "sunsize": "Размер солнца (мягкость), °", "rays": "Лучей на пиксель", "radius": "Радиус", "range": "Дальность",
        "taa": "Накопление по кадрам (TAA)", "taan": "Накоплено кадров (макс.)",
        "gimulti": "Многократные отскоки света",
        "light": "Огонь, свечи и лампы освещают всё вокруг", "lightstrength": "Яркость света", "lightmax": "Потолок яркости (против пересвета)", "lightrange": "Дальность света", "lightthr": "Порог яркости источника", "lightrays": "Теневых лучей на пиксель", "lightdebug": "Показать найденные источники (розовым)",
        "dyngeo": "Персонажи отбрасывают лучевые тени", "instgeo": "Мелкие инстансные объекты (эксперимент)",
        "aniso": "Анизотропия", "lodbias": "Резкость текстур (LOD bias)", "restart": "применяется при следующем запуске игры",
        "autosun": "Брать направление солнца из игры", "az": "Азимут, °", "el": "Высота, °",
        "applied": "Сохранено в SkyRT.cfg - игра подхватит изменения за секунду.",
        "folder": "Открыть папку", "refresh": "Обновить", "nolog": "Логов пока нет. Запусти игру с лучами.",
        "hotkeys": "В игре: Ctrl+Home - лучи вкл/выкл, Ctrl+End - режимы отладки",
        "lang": "English",
    },
    "en": {
        "title": "SkyRT - rays for Sky",
        "tab_game": "Game", "tab_gfx": "Graphics", "tab_log": "Log",
        "st_dll": "Mod files", "st_layer": "Vulkan layer", "st_game": "Game",
        "ok_dll": "found", "no_dll": "NOT FOUND (put SkyRT.dll and VK_LAYER_SKYRT.json next to the panel)",
        "layer_on": "installed", "layer_off": "not installed", "layer_na": "Windows only",
        "game_set": "set", "game_unset": "not set",
        "install": "Install ray tracing", "uninstall": "Uninstall",
        "game_path": "Path to Sky.exe", "browse": "Browse...",
        "play_rt": "▶  Play with rays", "play_plain": "Play without rays",
        "installed_ok": "Layer installed. Start the game with \"Play with rays\".",
        "uninstalled_ok": "Layer removed from the registry. You can delete the files manually.",
        "err": "Error", "no_game": "Set the path to Sky.exe first.", "no_layer": "Install ray tracing first.",
        "preset": "Preset", "p_low": "Low", "p_mid": "Medium", "p_high": "High", "p_ultra": "Ultra", "p_custom": "Custom",
        "g_main": "General", "g_sh": "Shadows", "g_ao": "Ambient occlusion (AO)", "g_gi": "Bounce light (GI)",
        "g_dn": "Denoising", "g_obj": "Objects", "g_tex": "Textures", "g_sun": "Sun", "g_light": "Light from fire and lamps",
        "rt_on": "Ray tracing enabled", "shadows": "Shadows", "ao": "AO", "gi": "GI",
        "strength": "Strength", "sunsize": "Sun size (softness), deg", "rays": "Rays per pixel", "radius": "Radius", "range": "Range",
        "taa": "Temporal accumulation (TAA)", "taan": "Accumulated frames (max)",
        "gimulti": "Multi-bounce light",
        "light": "Fire, candles and lamps light their surroundings", "lightstrength": "Light strength", "lightmax": "Brightness ceiling (no blow-out)", "lightrange": "Light range", "lightthr": "Source brightness threshold", "lightrays": "Shadow rays per pixel", "lightdebug": "Show detected sources (magenta)",
        "dyngeo": "Characters cast ray-traced shadows", "instgeo": "Small instanced props (experimental)",
        "aniso": "Anisotropic filtering", "lodbias": "Texture sharpness (LOD bias)", "restart": "applies on the next game start",
        "autosun": "Take the sun direction from the game", "az": "Azimuth, deg", "el": "Elevation, deg",
        "applied": "Saved to SkyRT.cfg - the game picks it up within a second.",
        "folder": "Open folder", "refresh": "Refresh", "nolog": "No logs yet. Start the game with rays.",
        "hotkeys": "In game: Ctrl+Home - rays on/off, Ctrl+End - debug views",
        "lang": "Русский",
    },
}

# ---------------------------------------------------------------- settings model
# key: (default, min, max, kind)   kind: b = bool, i = int, f = float
SETTINGS = {
    "enabled": (1, 0, 1, "b"),
    "shadows": (1, 0, 1, "b"), "strength": (0.5, 0.0, 1.0, "f"), "sunsize": (2.5, 0.0, 10.0, "f"), "shrays": (2, 1, 8, "i"),
    "ao": (1, 0, 1, "b"), "aostrength": (0.7, 0.0, 1.0, "f"), "aoradius": (1.5, 0.05, 20.0, "f"), "aorays": (4, 1, 8, "i"),
    "gi": (1, 0, 1, "b"), "gistrength": (0.35, 0.0, 2.0, "f"), "girange": (30.0, 1.0, 200.0, "f"),
    "taa": (1, 0, 1, "b"), "taan": (12.0, 1.0, 64.0, "f"),
    "dyngeo": (1, 0, 1, "b"), "instgeo": (1, 0, 1, "b"),
    "gimulti": (0.6, 0.0, 0.9, "f"),
    "light": (1, 0, 1, "b"), "lightstrength": (10.0, 0.0, 200.0, "f"), "lightmax": (0.6, 0.05, 5.0, "f"), "lightrange": (10.0, 1.0, 60.0, "f"), "lightthr": (2.0, 0.3, 50.0, "f"),
    "lightrays": (2, 1, 4, "i"), "lightdebug": (0, 0, 1, "b"),
    "aniso": (16, 0, 16, "i"), "lodbias": (0.0, -2.0, 1.0, "f"),
    "autosun": (1, 0, 1, "b"), "az": (30.0, 0.0, 360.0, "f"), "el": (50.0, 1.0, 89.0, "f"),
}
PRESETS = {
    "p_low":   {"shrays": 1, "aorays": 2, "taan": 8, "aniso": 4, "gi": 0},
    "p_mid":   {"shrays": 2, "aorays": 4, "taan": 12, "aniso": 8, "gi": 1},
    "p_high":  {"shrays": 4, "aorays": 6, "taan": 16, "aniso": 16, "gi": 1, "girange": 40.0},
    "p_ultra": {"shrays": 8, "aorays": 8, "taan": 24, "aniso": 16, "gi": 1, "girange": 60.0},
}


def app_dir():
    return os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))


def install_dir():
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, APP_NAME)


def panel_cfg_path():
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    d = os.path.join(base, APP_NAME)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "panel.json")


def find_source_files():
    """Folder that holds SkyRT.dll + manifest: next to the panel, or in ./layer."""
    for d in (app_dir(), os.path.join(app_dir(), "layer"), os.path.join(app_dir(), "dist")):
        if os.path.isfile(os.path.join(d, DLL)) and os.path.isfile(os.path.join(d, MANIFEST)):
            return d
    return None


def installed_manifest():
    return os.path.join(install_dir(), MANIFEST)


def layer_registered():
    if winreg is None:
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_PATH, 0, winreg.KEY_READ) as k:
            winreg.QueryValueEx(k, installed_manifest())
            return True
    except OSError:
        return False


def register_layer():
    os.makedirs(install_dir(), exist_ok=True)
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, REG_PATH, 0, winreg.KEY_SET_VALUE) as k:
        winreg.SetValueEx(k, installed_manifest(), 0, winreg.REG_DWORD, 0)


def unregister_layer():
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_PATH, 0, winreg.KEY_SET_VALUE) as k:
            winreg.DeleteValue(k, installed_manifest())
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------- cfg file
def read_cfg(path):
    vals, order = {}, []
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.split("#", 1)[0].strip()
                if "=" in s:
                    k, v = s.split("=", 1)
                    vals[k.strip()] = v.strip()
                    order.append(k.strip())
    return vals, order


def write_cfg(path, values):
    """Keeps unknown keys (water fingerprints etc.), writes compact lines: the layer reads at most 4 KB."""
    old, order = read_cfg(path)
    merged = dict(old)
    for k, v in values.items():
        merged[k] = v
    keys = [k for k in order if k in merged] + [k for k in merged if k not in order]
    lines = ["# written by SkyRT Panel - live settings, applied while the game runs"]
    lines += [f"{k}={merged[k]}" for k in keys]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)


def fmt(v, kind):
    if kind == "b":
        return str(int(bool(v)))
    if kind == "i":
        return str(int(v))
    return ("%.3f" % float(v)).rstrip("0").rstrip(".") or "0"


# ---------------------------------------------------------------- widgets
class FSlider(QWidget):
    """Slider + value label for a float/int range."""
    def __init__(self, lo, hi, kind, parent=None):
        super().__init__(parent)
        self.lo, self.hi, self.kind = lo, hi, kind
        self.steps = (hi - lo) if kind == "i" else 200
        self.s = QSlider(Qt.Horizontal)
        self.s.setRange(0, int(self.steps))
        self.l = QLabel()
        self.l.setMinimumWidth(48)
        self.l.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.s, 1)
        lay.addWidget(self.l)
        self.s.valueChanged.connect(self._upd)

    def _upd(self, _=None):
        self.l.setText(fmt(self.value(), self.kind))

    def value(self):
        v = self.lo + (self.hi - self.lo) * self.s.value() / self.steps
        return int(round(v)) if self.kind == "i" else v

    def setValue(self, v):
        self.s.blockSignals(True)
        self.s.setValue(int(round((float(v) - self.lo) / (self.hi - self.lo) * self.steps)))
        self.s.blockSignals(False)
        self._upd()

    def changed(self, fn):
        self.s.valueChanged.connect(fn)


class Panel(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cfgp = {}
        try:
            with open(panel_cfg_path(), "r", encoding="utf-8") as f:
                self.cfgp = json.load(f)
        except Exception:
            pass
        self.lang = self.cfgp.get("lang", "ru")
        self.widgets = {}
        self.loading = False
        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.timeout.connect(self.save_cfg)
        self.log_timer = QTimer(self)
        self.log_timer.timeout.connect(self.refresh_log)
        self.build()
        self.retranslate()
        self.refresh_status()
        self.load_cfg()
        self.log_timer.start(2000)

    # ------------------------------------------------------------ ui
    def t(self, k):
        return TXT[self.lang][k]

    def build(self):
        self.resize(640, 720)
        root = QWidget()
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        self.tabs = QTabWidget()
        v.addWidget(self.tabs, 1)
        self.hot = QLabel()
        self.hot.setStyleSheet("color:#8a8f98")
        v.addWidget(self.hot)
        self.msg = QLabel("")
        self.msg.setWordWrap(True)
        v.addWidget(self.msg)

        # --- game tab
        g = QWidget()
        gl = QVBoxLayout(g)
        self.st_box = QGroupBox()
        sf = QFormLayout(self.st_box)
        self.st_dll_l, self.st_layer_l, self.st_game_l = QLabel(), QLabel(), QLabel()
        self.lbl_dll, self.lbl_layer, self.lbl_game = QLabel(), QLabel(), QLabel()
        sf.addRow(self.lbl_dll, self.st_dll_l)
        sf.addRow(self.lbl_layer, self.st_layer_l)
        sf.addRow(self.lbl_game, self.st_game_l)
        gl.addWidget(self.st_box)
        row = QHBoxLayout()
        self.b_install, self.b_uninstall = QPushButton(), QPushButton()
        self.b_install.clicked.connect(self.do_install)
        self.b_uninstall.clicked.connect(self.do_uninstall)
        row.addWidget(self.b_install, 2)
        row.addWidget(self.b_uninstall, 1)
        gl.addLayout(row)
        self.lbl_path = QLabel()
        gl.addWidget(self.lbl_path)
        prow = QHBoxLayout()
        self.e_game = QLineEdit(self.cfgp.get("game", ""))
        self.e_game.textChanged.connect(self.game_changed)
        self.b_browse = QPushButton()
        self.b_browse.clicked.connect(self.browse)
        prow.addWidget(self.e_game, 1)
        prow.addWidget(self.b_browse)
        gl.addLayout(prow)
        self.b_play = QPushButton()
        self.b_play.setMinimumHeight(56)
        f = QFont(); f.setPointSize(13); f.setBold(True)
        self.b_play.setFont(f)
        self.b_play.clicked.connect(lambda: self.play(True))
        self.b_plain = QPushButton()
        self.b_plain.clicked.connect(lambda: self.play(False))
        gl.addWidget(self.b_play)
        gl.addWidget(self.b_plain)
        gl.addStretch(1)
        self.b_lang = QPushButton()
        self.b_lang.clicked.connect(self.toggle_lang)
        gl.addWidget(self.b_lang)
        self.tabs.addTab(g, "")

        # --- graphics tab
        x = QWidget()
        xl = QVBoxLayout(x)
        prow = QHBoxLayout()
        self.lbl_preset = QLabel()
        self.c_preset = QComboBox()
        self.c_preset.activated.connect(self.apply_preset)
        prow.addWidget(self.lbl_preset)
        prow.addWidget(self.c_preset, 1)
        xl.addLayout(prow)
        self.boxes = {}

        def group(key):
            b = QGroupBox()
            self.boxes[key] = b
            fl = QFormLayout(b)
            xl.addWidget(b)
            return fl

        def add_bool(fl, key, label_key):
            c = QCheckBox()
            self.widgets[key] = c
            c.stateChanged.connect(self.touched)
            fl.addRow(c)
            self.labels.append((c, label_key, True))

        def add_slider(fl, key, label_key):
            lo, hi, kind = SETTINGS[key][1], SETTINGS[key][2], SETTINGS[key][3]
            s = FSlider(lo, hi, kind)
            s.changed(self.touched)
            self.widgets[key] = s
            lab = QLabel()
            fl.addRow(lab, s)
            self.labels.append((lab, label_key, False))

        self.labels = []
        fl = group("g_main"); add_bool(fl, "enabled", "rt_on")
        fl = group("g_sh"); add_bool(fl, "shadows", "shadows"); add_slider(fl, "strength", "strength"); add_slider(fl, "sunsize", "sunsize"); add_slider(fl, "shrays", "rays")
        fl = group("g_ao"); add_bool(fl, "ao", "ao"); add_slider(fl, "aostrength", "strength"); add_slider(fl, "aoradius", "radius"); add_slider(fl, "aorays", "rays")
        fl = group("g_gi"); add_bool(fl, "gi", "gi"); add_slider(fl, "gistrength", "strength"); add_slider(fl, "girange", "range"); add_slider(fl, "gimulti", "gimulti")
        fl = group("g_light"); add_bool(fl, "light", "light"); add_slider(fl, "lightstrength", "lightstrength"); add_slider(fl, "lightmax", "lightmax"); add_slider(fl, "lightrange", "lightrange")
        add_slider(fl, "lightthr", "lightthr"); add_slider(fl, "lightrays", "lightrays"); add_bool(fl, "lightdebug", "lightdebug")
        fl = group("g_dn"); add_bool(fl, "taa", "taa"); add_slider(fl, "taan", "taan")
        fl = group("g_obj"); add_bool(fl, "dyngeo", "dyngeo"); add_bool(fl, "instgeo", "instgeo")
        fl = group("g_tex"); add_slider(fl, "aniso", "aniso"); add_slider(fl, "lodbias", "lodbias")
        self.restart_lbl = QLabel(); self.restart_lbl.setStyleSheet("color:#8a8f98")
        fl.addRow(self.restart_lbl)
        fl = group("g_sun"); add_bool(fl, "autosun", "autosun"); add_slider(fl, "az", "az"); add_slider(fl, "el", "el")
        xl.addStretch(1)
        from PyQt5.QtWidgets import QScrollArea
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setWidget(x); sc.setFrameShape(QScrollArea.NoFrame)
        self.tabs.addTab(sc, "")

        # --- log tab
        lg = QWidget()
        ll = QVBoxLayout(lg)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Consolas", 9))
        ll.addWidget(self.log_view, 1)
        lr = QHBoxLayout()
        self.b_refresh, self.b_folder = QPushButton(), QPushButton()
        self.b_refresh.clicked.connect(self.refresh_log)
        self.b_folder.clicked.connect(self.open_folder)
        lr.addWidget(self.b_refresh); lr.addWidget(self.b_folder); lr.addStretch(1)
        ll.addLayout(lr)
        self.tabs.addTab(lg, "")

    def retranslate(self):
        self.setWindowTitle(self.t("title"))
        for i, k in enumerate(("tab_game", "tab_gfx", "tab_log")):
            self.tabs.setTabText(i, self.t(k))
        self.st_box.setTitle("")
        self.lbl_dll.setText(self.t("st_dll")); self.lbl_layer.setText(self.t("st_layer")); self.lbl_game.setText(self.t("st_game"))
        self.b_install.setText(self.t("install")); self.b_uninstall.setText(self.t("uninstall"))
        self.lbl_path.setText(self.t("game_path")); self.b_browse.setText(self.t("browse"))
        self.b_play.setText(self.t("play_rt")); self.b_plain.setText(self.t("play_plain"))
        self.b_lang.setText(self.t("lang"))
        self.lbl_preset.setText(self.t("preset"))
        cur = self.c_preset.currentIndex()
        self.c_preset.clear()
        for k in ("p_low", "p_mid", "p_high", "p_ultra", "p_custom"):
            self.c_preset.addItem(self.t(k), k)
        self.c_preset.setCurrentIndex(cur if cur >= 0 else 4)
        for key, b in self.boxes.items():
            b.setTitle(self.t(key))
        for w, k, is_check in self.labels:
            w.setText(self.t(k))
        self.restart_lbl.setText("* " + self.t("aniso") + " / " + self.t("lodbias") + ": " + self.t("restart"))
        self.b_refresh.setText(self.t("refresh")); self.b_folder.setText(self.t("folder"))
        self.hot.setText(self.t("hotkeys"))
        self.refresh_status()

    def toggle_lang(self):
        self.lang = "en" if self.lang == "ru" else "ru"
        self.cfgp["lang"] = self.lang
        self.save_panel()
        self.retranslate()

    # ------------------------------------------------------------ status / install
    def status(self, label, text, ok):
        label.setText(text)
        label.setStyleSheet("color:%s" % ("#4cc38a" if ok else "#e5534b"))

    def refresh_status(self):
        have = find_source_files() is not None or os.path.isfile(os.path.join(install_dir(), DLL))
        self.status(self.st_dll_l, self.t("ok_dll") if have else self.t("no_dll"), have)
        reg = layer_registered()
        if reg is None:
            self.status(self.st_layer_l, self.t("layer_na"), False)
        else:
            self.status(self.st_layer_l, self.t("layer_on") if reg else self.t("layer_off"), reg)
        gp = self.e_game.text().strip().strip('"')
        ok = bool(gp) and os.path.isfile(gp)
        self.status(self.st_game_l, self.t("game_set") if ok else self.t("game_unset"), ok)

    def say(self, text, err=False):
        self.msg.setText(text)
        self.msg.setStyleSheet("color:%s" % ("#e5534b" if err else "#4cc38a"))

    def do_install(self):
        if winreg is None:
            QMessageBox.warning(self, self.t("err"), self.t("layer_na"))
            return
        src = find_source_files()
        try:
            if src:
                os.makedirs(install_dir(), exist_ok=True)
                for n in (DLL, MANIFEST):
                    s, d = os.path.join(src, n), os.path.join(install_dir(), n)
                    if os.path.abspath(s) != os.path.abspath(d):
                        shutil.copy2(s, d)
            elif not os.path.isfile(os.path.join(install_dir(), DLL)):
                QMessageBox.warning(self, self.t("err"), self.t("no_dll"))
                return
            register_layer()
            self.say(self.t("installed_ok"))
        except Exception as e:  # noqa
            QMessageBox.warning(self, self.t("err"), str(e))
        self.refresh_status()
        self.load_cfg()

    def do_uninstall(self):
        if winreg is None:
            return
        try:
            unregister_layer()
            self.say(self.t("uninstalled_ok"))
        except Exception as e:  # noqa
            QMessageBox.warning(self, self.t("err"), str(e))
        self.refresh_status()

    # ------------------------------------------------------------ game
    def browse(self):
        p, _ = QFileDialog.getOpenFileName(self, self.t("game_path"), self.e_game.text() or "", "Sky.exe (*.exe);;*.*")
        if p:
            self.e_game.setText(p)

    def game_changed(self, _=None):
        self.cfgp["game"] = self.e_game.text().strip().strip('"')
        self.save_panel()
        self.refresh_status()

    def play(self, rays):
        gp = self.e_game.text().strip().strip('"')
        if not gp or not os.path.isfile(gp):
            QMessageBox.information(self, self.t("err"), self.t("no_game"))
            return
        if rays and layer_registered() is False:
            QMessageBox.information(self, self.t("err"), self.t("no_layer"))
            return
        env = dict(os.environ)
        if rays:
            env["SKYRT_ENABLE"] = "1"
        else:
            env.pop("SKYRT_ENABLE", None)
        flags = 0
        if os.name == "nt":
            flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        try:
            subprocess.Popen([gp], cwd=os.path.dirname(gp), env=env, creationflags=flags, close_fds=True)
        except Exception as e:  # noqa
            QMessageBox.warning(self, self.t("err"), str(e))

    # ------------------------------------------------------------ cfg
    def cfg_path(self):
        d = install_dir()
        if not os.path.isdir(d):
            s = find_source_files()
            d = s or d
        return os.path.join(d, "SkyRT.cfg")

    def load_cfg(self):
        vals, _ = read_cfg(self.cfg_path())
        self.loading = True
        for k, (dflt, lo, hi, kind) in SETTINGS.items():
            w = self.widgets.get(k)
            if w is None:
                continue
            try:
                v = float(vals.get(k, dflt))
            except ValueError:
                v = dflt
            v = max(lo, min(hi, v))
            if isinstance(w, QCheckBox):
                w.setChecked(bool(round(v)))
            else:
                w.setValue(v)
        self.loading = False

    def current_values(self):
        out = {}
        for k, (dflt, lo, hi, kind) in SETTINGS.items():
            w = self.widgets.get(k)
            if w is None:
                continue
            v = int(w.isChecked()) if isinstance(w, QCheckBox) else w.value()
            out[k] = fmt(v, kind)
        return out

    def touched(self, *_):
        if self.loading:
            return
        self.c_preset.setCurrentIndex(self.c_preset.findData("p_custom"))
        self.save_timer.start(300)

    def save_cfg(self):
        try:
            os.makedirs(os.path.dirname(self.cfg_path()), exist_ok=True)
            write_cfg(self.cfg_path(), self.current_values())
            self.say(self.t("applied"))
        except Exception as e:  # noqa
            self.say(str(e), True)

    def apply_preset(self, idx):
        key = self.c_preset.itemData(idx)
        p = PRESETS.get(key)
        if not p:
            return
        self.loading = True
        for k, v in p.items():
            w = self.widgets.get(k)
            if w is None:
                continue
            if isinstance(w, QCheckBox):
                w.setChecked(bool(v))
            else:
                w.setValue(v)
        self.loading = False
        self.save_cfg()

    # ------------------------------------------------------------ log
    def log_dir(self):
        return os.path.dirname(self.cfg_path())

    def refresh_log(self):
        if self.tabs.currentIndex() != 2 and self.log_view.toPlainText():
            return
        files = sorted(glob.glob(os.path.join(self.log_dir(), "SkyRT_*.log")), key=os.path.getmtime)
        if not files:
            self.log_view.setPlainText(self.t("nolog"))
            return
        try:
            with open(files[-1], "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 60000))
                text = f.read().decode("utf-8", errors="replace")
        except OSError:
            return
        bar = self.log_view.verticalScrollBar()
        stick = bar.value() >= bar.maximum() - 4
        self.log_view.setPlainText(text)
        if stick:
            bar.setValue(bar.maximum())

    def open_folder(self):
        os.makedirs(self.log_dir(), exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.log_dir()))

    def save_panel(self):
        try:
            with open(panel_cfg_path(), "w", encoding="utf-8") as f:
                json.dump(self.cfgp, f, ensure_ascii=False, indent=1)
        except OSError:
            pass


def dark(app):
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.Window, QColor(30, 31, 34)); p.setColor(QPalette.WindowText, QColor(225, 227, 230))
    p.setColor(QPalette.Base, QColor(24, 25, 27)); p.setColor(QPalette.AlternateBase, QColor(36, 37, 41))
    p.setColor(QPalette.Text, QColor(225, 227, 230)); p.setColor(QPalette.Button, QColor(45, 47, 52))
    p.setColor(QPalette.ButtonText, QColor(225, 227, 230)); p.setColor(QPalette.Highlight, QColor(60, 130, 230))
    p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(p)


def main():
    app = QApplication(sys.argv)
    dark(app)
    w = Panel()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
