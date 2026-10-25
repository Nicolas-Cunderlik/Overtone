"""Custom title bar - replaces the native one, which on Windows 10 turns grey
whenever the window loses focus (the caption colour API is Windows 11 only)."""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from . import theme

TITLE_BAR_HEIGHT = 64  # matches Spotify's own title bar


def _button_style(hover, pressed):
    return f"""
QPushButton {{
    background-color: transparent; color: {theme.MUTED_TEXT}; border: none;
    font-family: "{theme.ICON_FONT}"; font-size: 10px;
}}
QPushButton:hover {{ background-color: {hover}; color: {theme.TEXT}; }}
QPushButton:pressed {{ background-color: {pressed}; }}
"""


class TitleBar(QWidget):
    def __init__(self, window, icon_path):
        super().__init__(window)
        self._window = window
        self.setFixedHeight(TITLE_BAR_HEIGHT)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(f"background-color: {theme.WINDOW_BG};")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(6)

        icon = QLabel()
        icon.setPixmap(QPixmap(icon_path).scaled(16, 16, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # Segoe MDL2 Assets glyph: ChromeClose
        close_btn = self._button("\ue8bb", "#e81123", "#f1707a")
        close_btn.clicked.connect(window.close)

        layout.addWidget(icon)
        layout.addStretch()
        layout.addWidget(close_btn)

    def _button(self, glyph, hover, pressed):
        btn = QPushButton(glyph)
        btn.setFixedSize(46, TITLE_BAR_HEIGHT)
        btn.setFocusPolicy(Qt.NoFocus)
        btn.setStyleSheet(_button_style(hover, pressed))
        return btn
