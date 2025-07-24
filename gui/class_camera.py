from PyQt6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QGroupBox, QFileDialog
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QObject, QRunnable, QPoint, pyqtSlot, QTimer
from PyQt6.QtGui import QPixmap, QImage
import numpy as np
import pylablib as pll
from pylablib.devices import DCAM
import pyqtgraph as pg
import time
from datetime import datetime
import tifffile as tf
import os

class CameraSignals(QObject):
    """Signals specific to camera operations."""
    frame_ready = pyqtSignal(np.ndarray)
    camera_error = pyqtSignal(str)
    camera_info = pyqtSignal(dict)  # For camera properties
    fps_update = pyqtSignal(float)
    resolution_changed = pyqtSignal(tuple)
    capture_saved = pyqtSignal(str)  # Path of saved image
    exposure_time_update = pyqtSignal(float)

class ImageViewerWithCursorInfo(pg.ImageView):
    """PyQT6 ImageView with added functionality to display cursor position and intensity."""

    cursor_position_changed = pyqtSignal(QPoint, float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Assurez-vous que l'histogramme est visible
        kwargs['view'] = pg.PlotItem()  # Pour garder les axes
        if 'hist' not in kwargs:
            kwargs['hist'] = True  # Active l'histogramme

        self.getView().scene().sigMouseMoved.connect(self.handle_mouse_move)
        self.mouse_pos = None
        self.value = None

        self.max_intensity = None  # Initialise l'intensité maximale
        self.max_intensity_label = QLabel("Max Intensity: N/A")  # Label pour afficher l'intensité max

    #     # Create a timer to update the cursor info even when the mouse is not moving
        self.update_timer = QTimer()
        self.update_timer.setInterval(100)  # Update every 100ms (0.1 seconds)

        self.update_timer.timeout.connect(self.update_max_intensity)
        # self.update_timer.timeout.connect(self.update_cursor_info)

        self.update_timer.start()

        # Configuration de l'histogramme
        self.ui.histogram.gradient.loadPreset('viridis')  # Optionnel : change le gradient de couleur
        # Pour ajuster la taille de l'histogramme
        self.ui.histogram.setMinimumWidth(40)  # Ajustez selon vos besoins


    def handle_mouse_move(self, pos):
        """Update the cursor position and intensity label when the mouse moves."""
        item = self.getImageItem()
        if item is None or item.image is None:
            return

        self.mouse_pos = self.getView().mapSceneToView(pos)
        x, y = int(self.mouse_pos.x()), int(self.mouse_pos.y())
        try:
            self.value = item.image[x, y]
        except IndexError:
            return

        self.cursor_position_changed.emit(QPoint(x, y), self.value)

    def update_max_intensity(self):
        if self.image is not None:
            self.max_intensity = np.max(self.image)  # Récupère l'intensité max
            self.max_intensity_label.setText(f"Max Intensity: {self.max_intensity:.2f}")  # Mise à jour du QLabel



class CAMWorker(QRunnable):
    def __init__(self, camera, savefolder="C:/Users/biof/Desktop/chloe/surgele"):
        super().__init__()
        self.signals = CameraSignals()
        self._running = False
        self._paused = False  # New flag for pause state
        self.camera = camera

        self.exposure = 1e-3
        self.gain = 1
        self.frame = None
        self.frame_buffer = []
        self._movie_acquisition = False
        self.tiff_writer = None
        self.savefolder = savefolder
        self.image_name = "snap"

        if hasattr(self.camera, 'setup_acquisition'):
            self.camera.setup_acquisition(mode="sequence", nframes=100)
        else:
            pass


    def set_exposure(self, exposure):
        """Dynamically set exposure while camera is running."""
        self.exposure = exposure * 1e-3
        if self.camera is not None:
            was_running = not self._paused
            if was_running:
                self._paused = True
                self.camera.stop_acquisition()

            self.camera.set_exposure(self.exposure)

            if was_running:
                self.camera.start_acquisition()
                self._paused = False


    def get_current_frame(self):
        """Return the latest frame if available."""
        return self.frame


    def update_savefolder(self, savefolder):
        self.savefolder = savefolder
        print(self.savefolder)

    def update_image_name(self, image_name):
        self.image_name = image_name
        print(self.image_name)

    def update_ROI(self, xmin, xmax, ymin, ymax, xbin, ybin):
        self.camera.set_roi(xmin,xmax,ymin,ymax, xbin, ybin)
        if hasattr(self.camera, "setup_acquisition"):
            self.camera.setup_acquisition(mode="sequence", nframes=100)
        else:
            pass

    def toggle_movie_acquisition(self, state):
        """Toggle movie acquisition and handle opening/closing TIFF files."""
        if state and not self._movie_acquisition:
            # Start recording to a new TIFF file with a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.filename = os.path.join(self.savefolder, f"camera_frames_{timestamp}.tiff")
            self.tiff_writer = tf.TiffWriter(self.filename, bigtiff=True)
            print(f"Started recording to {self.filename}")
        elif not state and self._movie_acquisition:
            # Stop recording and close the TIFF file
            self._movie_acquisition = state  # Mettre à False d'abord
            # Attendre un peu pour être sûr que plus aucune écriture n'est en cours
            time.sleep(0.05)

            if self.tiff_writer is not None:
                self.tiff_writer.close()
                self.tiff_writer = None
                print("Stopped recording")



        self._movie_acquisition = state

    def take_snap(self, path=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # if self._running:
        #     filename = os.path.join(self.savefolder, f"{self.image_name}_{timestamp}.tiff")
        #     tf.imwrite(filename, self.frame)
        # else:
        #     print("Start preview first")
        if self.frame is not None:
            if path:
                filename = path
            else:
                filename = os.path.join(self.savefolder, f"{self.image_name}_{timestamp}.tiff")

            # Ensure the frame is in uint16 format for saving as a 16-bit TIFF
            if self.frame.dtype != np.uint16:
                self.frame = self.frame.astype(np.uint16)

            # Save the frame as a 16-bit TIFF file
            tf.imwrite(filename, self.frame)
            print(f"Snapshot saved to {filename}")
        else:
            print("No frame available to capture.")


    @pyqtSlot()
    def run(self):
        self._running = True
        try:
            self.camera.start_acquisition()
            last_display_time = time.time()  # Variable pour limiter la fréquence d'affichage
            display_interval = 1 / 60  # Limite d'affichage à 60 FPS (à ajuster)
            while self._running:
                if not self._paused:
                    try:
                        if hasattr(self.camera, "wait_for_frame"):
                            self.camera.wait_for_frame()
                        else:
                            pass
                        frame = self.camera.read_oldest_image()
                        if frame is not None:
                            self.frame = frame
                            # Write frame to TIFF if movie acquisition is active
                            if self._movie_acquisition and self.tiff_writer is not None:
                                try:
                                    self.tiff_writer.write(self.frame, contiguous=True)
                                except Exception as e:
                                    print(f"Erreur d'écriture TIFF: {str(e)}")

                            # Émettre des signaux seulement si l'intervalle est respecté
                            current_time = time.time()
                            if current_time - last_display_time >= display_interval:
                                self.signals.frame_ready.emit(self.frame)
                                # self.signals.exposure_time_update.emit(self.camera.get_attribute_value("EXPOSURE_TIME"))
                                self.signals.fps_update.emit(1 / self.camera.get_frame_period())
                                last_display_time = current_time


                    except Exception as e:
                        print(f"Error reading frame: {str(e)}")
                else:
                    time.sleep(0.01)
        except Exception as e:
            self.signals.camera_error.emit(f"PVCAM capture error: {str(e)}")
        finally:
            if self.camera is not None:
                self.camera.stop_acquisition()
            if self.tiff_writer is not None:
                self.tiff_writer.close()  # Close any open TIFF writer at the end

    def stop(self):
        """Stop PVCAM camera capture."""
        self._running = False


