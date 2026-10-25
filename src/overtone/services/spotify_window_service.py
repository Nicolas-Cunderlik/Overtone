"""Locates the native Spotify desktop window via the Win32 API so the GUI can
dock itself against it. Windows-only, matches by owning process name (Spotify
renames its window title to the current track, so title matching isn't reliable).
"""

import ctypes
import ntpath
from ctypes import wintypes

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
GW_OWNER = 4

_EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


def _process_name(pid):
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ""
    try:
        # QueryFullProcessImageNameW works with limited-query access; GetModuleBaseNameW
        # needs PROCESS_VM_READ and fails for packaged (Microsoft Store) Spotify.
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(len(buf))
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return ""
        return ntpath.basename(buf.value)
    finally:
        kernel32.CloseHandle(handle)


def find_spotify_window():
    """Returns the HWND of Spotify's main top-level window, or None if it isn't running."""
    found = []

    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd) or user32.GetWindow(hwnd, GW_OWNER):
            return True
        if user32.GetWindowTextLengthW(hwnd) == 0:
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if _process_name(pid.value).lower() != "spotify.exe":
            return True
        found.append(hwnd)
        return False

    user32.EnumWindows(_EnumWindowsProc(callback), 0)
    return found[0] if found else None


def is_window_valid(hwnd):
    return bool(user32.IsWindow(hwnd))


def is_minimized(hwnd):
    return bool(user32.IsIconic(hwnd))


def get_window_rect(hwnd):
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    return rect.left, rect.top, rect.right, rect.bottom


def get_foreground_window():
    return user32.GetForegroundWindow()


# --- Event-driven tracking -------------------------------------------------
# SetWinEventHook delivers a callback every time the target window moves
# (including each step of a drag), minimises, restores, gains focus or closes,
# so docking stays in lock-step without polling.

dwmapi = ctypes.windll.dwmapi

EVENT_SYSTEM_FOREGROUND = 0x0003
EVENT_SYSTEM_MINIMIZESTART = 0x0016
EVENT_SYSTEM_MINIMIZEEND = 0x0017
EVENT_OBJECT_DESTROY = 0x8001
EVENT_OBJECT_HIDE = 0x8003
EVENT_OBJECT_LOCATIONCHANGE = 0x800B
OBJID_WINDOW = 0

WINEVENT_OUTOFCONTEXT = 0x0000
WINEVENT_SKIPOWNPROCESS = 0x0002

DWMWA_EXTENDED_FRAME_BOUNDS = 9

SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
HWND_TOP = 0

_WinEventProc = ctypes.WINFUNCTYPE(
    None,
    wintypes.HANDLE,
    wintypes.DWORD,
    wintypes.HWND,
    wintypes.LONG,
    wintypes.LONG,
    wintypes.DWORD,
    wintypes.DWORD,
)

user32.SetWinEventHook.argtypes = [
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.HMODULE,
    _WinEventProc,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
]
user32.SetWinEventHook.restype = wintypes.HANDLE
user32.UnhookWinEvent.argtypes = [wintypes.HANDLE]
user32.SetWindowPos.argtypes = [
    wintypes.HWND,
    wintypes.HWND,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.UINT,
]
dwmapi.DwmGetWindowAttribute.argtypes = [
    wintypes.HWND,
    wintypes.DWORD,
    ctypes.c_void_p,
    wintypes.DWORD,
]


def get_visible_rect(hwnd):
    """Window bounds as drawn on screen. Unlike GetWindowRect this excludes the
    invisible resize borders Windows 10 adds, so docked windows sit flush."""
    rect = wintypes.RECT()
    if dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(rect), ctypes.sizeof(rect)) != 0:
        return get_window_rect(hwnd)
    return rect.left, rect.top, rect.right, rect.bottom


def set_window_bounds(hwnd, x, y, width, height):
    """Moves and resizes without re-ordering or stealing focus (physical pixels)."""
    user32.SetWindowPos(hwnd, None, x, y, width, height, SWP_NOZORDER | SWP_NOACTIVATE)


def bring_to_top(hwnd):
    user32.SetWindowPos(hwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)


def place_below(hwnd, above_hwnd):
    """Moves `hwnd` in z-order to sit directly beneath `above_hwnd`, without activating it."""
    user32.SetWindowPos(hwnd, above_hwnd, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)


class WindowEventHook:
    """Calls on_event(event) for top-level events on `hwnd`. Must be created on a
    thread running a message loop (the Qt main thread) - callbacks arrive there."""

    _EVENT_RANGES = [
        (EVENT_SYSTEM_FOREGROUND, EVENT_SYSTEM_FOREGROUND),
        (EVENT_SYSTEM_MINIMIZESTART, EVENT_SYSTEM_MINIMIZEEND),
        (EVENT_OBJECT_DESTROY, EVENT_OBJECT_HIDE),
        (EVENT_OBJECT_LOCATIONCHANGE, EVENT_OBJECT_LOCATIONCHANGE),
    ]

    def __init__(self, hwnd, on_event):
        self.hwnd = hwnd
        self._on_event = on_event
        self._proc = _WinEventProc(self._callback)  # keep a reference or it gets GC'd
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        self._hooks = [
            h
            for h in (
                user32.SetWinEventHook(
                    lo, hi, None, self._proc, pid.value, 0, WINEVENT_OUTOFCONTEXT | WINEVENT_SKIPOWNPROCESS
                )
                for lo, hi in self._EVENT_RANGES
            )
            if h
        ]

    def _callback(self, _hook, event, hwnd, id_object, id_child, _thread, _time):
        # Hooks are per-process, so filter out child windows/accessibility objects
        if hwnd != self.hwnd or id_object != OBJID_WINDOW or id_child != 0:
            return
        try:
            self._on_event(event)
        except Exception:
            import logging

            logging.getLogger(__name__).exception("Window event handler failed")

    def unhook(self):
        for h in self._hooks:
            user32.UnhookWinEvent(h)
        self._hooks = []
