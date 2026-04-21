#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
import importlib.util
import site

# Linux-only platform selection. Do not force xcb/wayland on macOS/Windows.
if sys.platform.startswith("linux"):
    if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
        os.environ.setdefault("QT_QPA_PLATFORM", "wayland")
    else:
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")


def _candidate_plugin_paths() -> list[Path]:
    paths: list[Path] = []

    spec = importlib.util.find_spec("PyQt5")
    if spec and spec.submodule_search_locations:
        base = Path(next(iter(spec.submodule_search_locations)))
        paths.append(base / "Qt5" / "plugins" / "platforms")

    paths.append(Path("/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"))
    paths.append(Path("/usr/lib/qt5/plugins/platforms"))
    return paths


if sys.platform.startswith("linux") and not os.environ.get(
    "QT_QPA_PLATFORM_PLUGIN_PATH"
):
    for p in _candidate_plugin_paths():
        if p.exists():
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(p)
            os.environ.setdefault("QT_PLUGIN_PATH", str(p.parent))
            break

# Avoid loading incompatible user-site wheels ahead of system packages.
user_site = site.getusersitepackages()
if isinstance(user_site, str) and user_site in sys.path:
    sys.path.remove(user_site)
try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        from PySide6.QtWidgets import QApplication

try:
    from gui import MainWindow
except ModuleNotFoundError as exc:
    if exc.name != "gui":
        raise
    from pyqt_app.gui import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    if hasattr(app, "exec"):
        return app.exec()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
