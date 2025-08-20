import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QSlider, QGroupBox,QFileDialog, QSpinBox, QComboBox, QGraphicsView, QLineEdit, QButtonGroup, QRadioButton
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QObject, QRunnable
from PyQt6.QtGui import QPixmap, QImage

import numpy as np
import os
import skimage.io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy import ndimage
from PIL import Image
from scipy.ndimage import shift
from skimage.measure import regionprops, label
from tifffile import imwrite
import tifffile as tiff
import textwrap
import cv2
# from pymmcore import CMMCore
import re
import logging
import serial
from serial.tools import list_ports
from serial.serialutil import SerialException
import time
import sys
import io
# import qdarkstyle
# import hid
import pylablib as pll
from pylablib.devices import DCAM
import new_hilo1
import fourierslice_refocusing
from homemade_hilo import hilo_reconstruction, hilo_reconstruction_contrasted
from class_camera import CAMWorker, ImageViewerWithCursorInfo
from class_lfm import LFM
from pycobolt import CoboltLaser
from gui import LFMControlGUI
from dlpc350 import DLPC350
from class_imagestackviewer import ImageStackViewer



mm_dir = "C:/Program Files/Micro-Manager-2.0/"
save_dir = "D:/surgele"
# save_dir = "E:/surgele"
# save_dir = "C:/Users/biof/Documents/Light Field Microscopy/surgele"

app = QApplication(sys.argv)

dark_style = """
    QWidget {
        background-color: #2E2E2E;
        color: #FFFFFF;
        font-size: 12px;
    }
    QPushButton {
        background-color: #444;
        color: white;
    }
    QPushButton:hover {
        background-color: '#555';
    }
    QPushButton:pressed {
        background-color: 'royal blue';
    }
    QLineEdit {
        background-color: #3A3A3A;
        color: white;
    }
    QLabel {
        color: white;
    }
    QGroupBox {
        border: 2px solid white;  # White border
        border-radius: 5px;        # Optional: round the corners
        padding: 10px;             # Optional: space between border and content
    }
"""
# app.setStyleSheet(dark_style)


mainWin = LFMControlGUI(mm_dir, save_dir)
mainWin.show()
sys.exit(app.exec())
