"""
Filename: overtone/ui/fullscreen_image.py
Author: Nicolas Cunderlik
Date: 2025
Description: Fullscreen album-cover window (Esc or Space closes it).

Copyright © 2025 Nicolas Cunderlik. All Rights Reserved.

This source code is the property of the author/owner and is protected under
copyright law. Unauthorized copying, modification, distribution, or any use
of this file, in part or in full, without prior written permission from the
author/owner, is strictly prohibited.

For inquiries, contact: nicolas7cunderlik@gmail.com
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget


class FullscreenImageWindow(QWidget):
    def __init__(self, data):
        super().__init__()

        # Set up the fullscreen window
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)  # Remove window decorations
        self.setStyleSheet("background-color: black;")
        self.showFullScreen()

        # Create a label to display the image
        self.image_label = QLabel(self)
        pixmap = QPixmap(1000, 1000)
        pixmap.loadFromData(data)
        self.image_label.setPixmap(
            pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Set layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)

    def keyPressEvent(self, a0):
        if a0 is not None and a0.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Space):
            self.close()  # Close the fullscreen window
