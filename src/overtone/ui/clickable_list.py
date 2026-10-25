"""An analytics box with a header and up to three clickable rows - used for
"Similar Tracks" (click plays) and "AI Chord Suggestions" (click downloads
MIDI). Each row carries a `key`; clicking emits `item_clicked(key)`. Rows
without a key are placeholders and can't be clicked. The hover tooltip (shown
after Qt's usual hover delay) explains what a click does."""

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QFontMetrics
from PyQt5.QtWidgets import QFrame, QLabel, QVBoxLayout

from . import theme

MAX_ROWS = 3
FLASH_MS = 2500

_ROW_STYLE = f"""
QLabel {{ color: {theme.TEXT}; font-size: 10px; border-radius: 4px; padding: 2px; }}
QLabel[clickable="true"]:hover {{ background-color: {theme.HOVER_BG}; }}
"""


class _Row(QLabel):
    clicked = pyqtSignal(object)

    def __init__(self, font):
        super().__init__()
        self.key = None
        self._text = ""
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignCenter)
        self.setAttribute(Qt.WA_Hover)
        if font:
            self.setFont(font)
        self.setStyleSheet(_ROW_STYLE)
        self._set_clickable(False)

    def set_item(self, key, text, tooltip):
        self.key = key
        self._show(self._fit_two_lines(text))
        self.setToolTip(tooltip)
        self._set_clickable(True)

    def set_placeholder(self, text):
        self.key = None
        self._show(text)
        self.setToolTip("")
        self._set_clickable(False)

    def flash(self, text):
        """Temporarily replaces the row's text, e.g. with a confirmation or error."""
        restore = self._text
        self.setText(text)
        QTimer.singleShot(FLASH_MS, lambda: self.setText(restore) if self.text() == text else None)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.key is not None:
            self.clicked.emit(self.key)
        super().mousePressEvent(event)

    def _show(self, text):
        self._text = text
        self.setText(text)

    def _set_clickable(self, clickable):
        self.setProperty("clickable", clickable)
        self.setCursor(Qt.PointingHandCursor if clickable else Qt.ArrowCursor)
        self.style().unpolish(self)  # re-evaluate the [clickable] style selector
        self.style().polish(self)

    def _fit_two_lines(self, text):
        # Rows word-wrap; cap at roughly two lines so three rows always fit
        # the fixed-height box. The full text stays available in the tooltip.
        width = self.width() if self.width() > 50 else 100
        return QFontMetrics(self.font()).elidedText(text, Qt.ElideRight, int(width * 1.8))


class ClickableListBox(QFrame):
    item_clicked = pyqtSignal(object)

    def __init__(self, title, font_family, width, height):
        super().__init__()
        self.setFixedSize(width, height)
        self.setStyleSheet(f"background-color: {theme.BOX_BG}; border-radius: 8px;")
        font = QFont(font_family, 10) if font_family else None

        self.header = QLabel(title)
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setStyleSheet(f"color: {theme.TEXT}; font-size: 10px;")
        if font:
            self.header.setFont(font)

        self.rows = [_Row(font) for _ in range(MAX_ROWS)]
        for row in self.rows:
            row.clicked.connect(self.item_clicked)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 9, 6, 9)
        layout.setSpacing(2)
        layout.addStretch(1)
        layout.addWidget(self.header)
        layout.addSpacing(8)
        for row in self.rows:
            layout.addWidget(row)
        layout.addStretch(1)

        self.set_idle()

    def set_items(self, items):
        """`items`: list of (key, text, tooltip); unused rows show "-"."""
        for i, row in enumerate(self.rows):
            if i < len(items):
                row.set_item(*items[i])
            else:
                row.set_placeholder("-")

    def set_placeholders(self, texts):
        for row, text in zip(self.rows, list(texts) + [""] * MAX_ROWS, strict=False):
            row.set_placeholder(text)

    def set_loading(self):
        self.set_placeholders(["loading..."])

    def set_idle(self):
        self.set_placeholders(["-", "-", "-"])

    def flash(self, key, text):
        for row in self.rows:
            if row.key == key:
                row.flash(text)
