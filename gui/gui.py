import sys
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QSlider, QGroupBox, QFileDialog, QSpinBox, QComboBox, QGraphicsView, QLineEdit, QButtonGroup, QRadioButton
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QThreadPool
from PyQt6.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pylablib as pll
from pylablib.devices import DCAM
import thorlabs_apt as apt

from class_camera import CAMWorker, ImageViewerWithCursorInfo
from class_lfm import LFM, LFMWorker
import pycobolt
from dlpc350 import DLPC350
from class_imagestackviewer import ImageStackViewer

class LFMControlGUI(QMainWindow):
    def __init__(self, mm_dir, save_dir):
        super().__init__()
        self.setWindowTitle("GUY")
        self.setGeometry(100, 100, 1200, 800)

        self.mm_dir = mm_dir
        self.save_dir = save_dir

        self.threads = []

        self.dlp_path = b'\\\\?\\HID#VID_0451&PID_6401&MI_00#7&3703db20&0&0000#{4d1e55b2-f16f-11cf-88cb-001111000030}'
        self.dlp = DLPC350(self.dlp_path, debug=1)
        self.dlp.connectDLP()
        self.dlp.enterStandby()
        self.lfm = LFM(self.mm_dir, self.save_dir)
        self.laser = pycobolt.CoboltLaser(port="COM3") # Creates a new Cobolt Laser object. Replace COMXX with the com-port of the laser
        self.laser.is_connected()
        self.laser.turn_off()

        self.camera = DCAM.DCAMCamera()
        self.camera.open()
        self.threadpool = QThreadPool()
        self.video_worker = None
        
        apt.list_available_devices() 
        self.motorx = apt.Motor(27256999)

        self.initUI()





    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QGridLayout()
        central_widget.setLayout(layout)


        ################## Camera Live View #################
        #####################################################

        camera_group = QGroupBox("Camera View")
        
        # self.camera_view = QLabel()
        # self.camera_view.setFixedSize(400, 400)

        # #Kinetix display
        self.live_display = QWidget()
        view_layout = QVBoxLayout()
        info_layout = QHBoxLayout()
        self.pos_info_label = QLabel("X  |  Y  |  I")

        self.live_display.setLayout(view_layout)
        
        self.image_view = ImageViewerWithCursorInfo(parent=self.live_display)
        self.image_view.cursor_position_changed.connect(self.handle_cursor_position)

        info_layout.addWidget(self.pos_info_label)
        info_layout.addWidget(self.image_view.max_intensity_label)
        view_layout.addWidget(self.image_view)
        view_layout.addLayout(info_layout)
        
        camera_group.setLayout(view_layout)
        
        

        ############## LFM Reconstruction View ###############
        ######################################################
 
        # GroupBox principal
        lfm_group = QGroupBox("LFM Reconstruction")
        lfm_layout = QHBoxLayout()  # Principal layout
        # Layouts for both images (righ and left)
        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()
        # Labels for image places
        self.left_image_label = QLabel("[Image]")
        self.right_image_label = QLabel("[Image]")
        # Image places sizes
        self.left_image_label.setFixedSize(250, 250)
        self.right_image_label.setFixedSize(250, 250)
        # file paths text lines
        self.left_image_path = QLineEdit(self)
        self.left_image_path.setPlaceholderText("Enter file path")
        self.right_image_path = QLineEdit(self)
        self.right_image_path.setPlaceholderText("Enter file path")
        # Buttons
        left_browse_button = QPushButton("Browse")
        right_browse_button = QPushButton("Browse")
        left_visualize_button = QPushButton("Visualize")
        right_visualize_button = QPushButton("Visualize")
        # Add widgets
        left_buttons_layout = QHBoxLayout()
        left_buttons_layout.addWidget(self.left_image_path)
        left_buttons_layout.addWidget(left_browse_button)
        right_buttons_layout = QHBoxLayout()
        right_buttons_layout.addWidget(self.right_image_path)
        right_buttons_layout.addWidget(right_browse_button)
        
        # Connexion buttons and methods
        left_browse_button.clicked.connect(lambda: self.browse_file(self.left_image_path))
        right_browse_button.clicked.connect(lambda: self.browse_file(self.right_image_path))
        left_visualize_button.clicked.connect(lambda: self.visualize_image(self.left_layout, self.left_image_path))
        right_visualize_button.clicked.connect(lambda: self.visualize_image(self.right_layout, self.right_image_path))
        
        # Add the layouts to principal layout
        self.left_layout.addWidget(self.left_image_label)
        self.left_layout.addLayout(left_buttons_layout)
        self.left_layout.addWidget(left_visualize_button)
        self.right_layout.addWidget(self.right_image_label)
        self.right_layout.addLayout(right_buttons_layout)
        self.right_layout.addWidget(right_visualize_button)
        lfm_layout.addLayout(self.left_layout)
        lfm_layout.addLayout(self.right_layout)
        lfm_group.setLayout(lfm_layout)
        
        ################ Camera parameters #################
        ######################################################

        cameraparam_group = QGroupBox("Camera Parameters")
        camera_layout = QVBoxLayout()
        
        
        expo_layout = QHBoxLayout()
        self.exposure_slider = QSlider(Qt.Orientation.Horizontal)
        self.exposure_slider.setMinimum(1)
        self.exposure_slider.setMaximum(1000) 
        self.exposure_slider.setValue(100)  # Initial value
        self.exposure_input = QLineEdit()
        self.exposure_input.setPlaceholderText("Enter exposure")
        self.exposure_input.setText(str(self.exposure_slider.value()))  # Init with slider value
        self.exposure_slider.valueChanged.connect(self.update_exposure_from_slider)
        self.exposure_input.returnPressed.connect(self.update_exposure_from_input)
        self.exposure_input.setFixedWidth(50)

        axial_layout = QHBoxLayout()
        self.axial_slider = QSlider(Qt.Orientation.Horizontal)
        self.axial_slider.setMinimum(4000) # 4mm c'est déja assez loin de l'échantillon
        self.axial_slider.setMaximum(5800) # 5.8mm c'est la limite apres ca touche l'echantillon
        self.axial_slider.setValue(4000) 
        self.axial_input = QLineEdit()
        self.axial_input.setPlaceholderText("Enter axial displacement")
        self.axial_input.setText(str(self.axial_slider.value()))
        self.axial_slider.valueChanged.connect(self.update_axial_from_slider)
        self.axial_input.returnPressed.connect(self.update_axial_from_input)
        self.axial_input.setFixedWidth(50)
    
        control_layout = QHBoxLayout()
        self.live_button = QPushButton("Live")
        self.live_button.setCheckable(True)
        self.live_button.clicked.connect(lambda: self.toggle_video(self.live_button))
        
        self.save_snapshot_button = QPushButton("Save Snapshot")
        self.save_snapshot_button.clicked.connect(lambda: self.take_snapshot())

        control_layout.addWidget(self.live_button)
        control_layout.addWidget(self.save_snapshot_button)
        axial_layout.addWidget(QLabel("Axial displacement:"))
        axial_layout.addWidget(self.axial_slider,1)
        axial_layout.addWidget(self.axial_input,0)
        expo_layout.addWidget(QLabel("Exposure:"))
        expo_layout.addWidget(self.exposure_slider,1)
        expo_layout.addWidget(self.exposure_input,0)
        
        camera_layout.addLayout(axial_layout)
        camera_layout.addLayout(expo_layout)
        camera_layout.addWidget(self.live_display)
        camera_layout.addLayout(control_layout)
        
        cameraparam_group.setLayout(camera_layout)
        
        #################### Laser Control ###################
        ######################################################

        laser_group = QGroupBox("Laser Control")
        laser_layout = QVBoxLayout()
        power_layout = QHBoxLayout()
        self.power_slider = QSlider(Qt.Orientation.Horizontal)
        self.power_slider.setMinimum(0)
        self.power_slider.setMaximum(70) 
        self.power_slider.setValue(1)
        self.power_input = QLineEdit()
        self.power_input.setPlaceholderText("Enter power value")
        self.power_input.setText(str(self.power_slider.value()))
        self.power_slider.valueChanged.connect(self.update_power_from_slider)
        self.power_input.returnPressed.connect(self.update_power_from_input)
        laser_toggle_button = QPushButton("OFF")
        laser_toggle_button.setCheckable(True)
        laser_toggle_button.clicked.connect(lambda: self.toggle_button_laser(laser_toggle_button, self.laser))
        
        self.power_input.setFixedWidth(50)

        power_layout.addWidget(QLabel("Power:"))
        power_layout.addWidget(self.power_slider,1)
        power_layout.addWidget(self.power_input,0)
        laser_layout.addLayout(power_layout)
        laser_layout.addWidget(laser_toggle_button)
        laser_group.setLayout(laser_layout)
        
        ################### DMD Control #####################
        #####################################################

        dmd_group = QGroupBox("DMD Control")
        dmd_layout = QVBoxLayout()
        # video_mode_button = QRadioButton("Video Mode")
        # display_mode_button = QRadioButton("Display Mode")
        # mode_group = QButtonGroup()
        # mode_group.addButton(video_mode_button)
        # mode_group.addButton(display_mode_button)

        # video_mode_button.clicked.connect(lambda: self.dlp.setDisplayMode(mode=0))
        # display_mode_button.clicked.connect(lambda: self.dlp.setDisplayMode(mode=1))
        # self.dlp.setDisplayMode(mode=0)
        
        selection = QHBoxLayout()
        # input_source_selector = QComboBox()
        # input_source_selector.addItems(["Flash", "Internal Test Patterns"])
        # input_source_selector.currentTextChanged.connect(lambda: self.handle_input_source_change)
        # self.dlp.setInputSource(source=2)
        image_selector = QComboBox()
        image_selector.addItems(["Logo TI", "Check1", "Check2", "Plain", "Lines1", "Lines2", "Lines3", "Lines4", "Speckle1", "Speckle2", "Speckle3"]) # pour avoir mes images des indices 0 à 8
        # image_selector.currentTextChanged.connect(lambda: self.dlp.loadImage(index=int(image_selector.currentText())))
        image_selector.currentIndexChanged.connect(lambda index: self.dlp.loadImage(index=index))
        # selection.addWidget(input_source_selector)
        selection.addWidget(image_selector)

        standby_button = QPushButton("OFF")
        standby_button.setCheckable(True)
        standby_button.clicked.connect(lambda: self.toggle_button_dmd(standby_button, self.dlp))

        dmd_layout.addWidget(QLabel("Video Mode"))
        # dmd_layout.addWidget(video_mode_button)
        # dmd_layout.addWidget(display_mode_button)
        # dmd_layout.addWidget(QLabel("Select Input Source:"))
        dmd_layout.addWidget(QLabel("Select Flash Index:"))
        dmd_layout.addLayout(selection)
        dmd_layout.addWidget(standby_button)
        dmd_group.setLayout(dmd_layout)
        
        ########### LFM Reconstruction Parameters ###########
        #####################################################

        lfm_param_group = QGroupBox("LFM Reconstruction Parameters")
        lfm_param_layout = QVBoxLayout()
        self.file_paths = ["", "", "", "", ""]  # List to store the 5 file paths
        files_labels = ["Reference:", "Background:", "Uniform:", "Pattern:", "Patterns Dir:"]
        for i in range(5):
            row_layout = QHBoxLayout()  # Horizontal layout for each row
            label = QLabel(files_labels[i])
            file_input = QLineEdit(self)
            file_input.setPlaceholderText(f"Enter path")
            if i==4:
                browse_button = QPushButton("Browse", self)
                browse_button.adjustSize()
                browse_button.clicked.connect(lambda _, f=file_input, idx=i: self.browse_dir(f, idx))  
            else:
                browse_button = QPushButton("Browse", self)
                browse_button.adjustSize()
                browse_button.clicked.connect(lambda _, f=file_input, idx=i: self.browse_image(f, idx))  # Pass index to update list
                save_button = QPushButton("Snap", self)
                save_button.adjustSize()
                save_button.clicked.connect(lambda _, f=file_input, idx=i: self.save_snap(f, idx)) # put fct to save snap shot

            row_layout.addWidget(label) # Add widgets to the row
            row_layout.addWidget(file_input)
            row_layout.addWidget(browse_button)
            row_layout.addWidget(save_button)
            # self.file_paths.append(file_input.text()) # Store the QLineEdit for later use
            lfm_param_layout.addLayout(row_layout) # Add the row to the main layout

        hilo_button = QRadioButton("HiLo-LFM")
        basic_button = QRadioButton("Basic LFM")
        recon_mode_layout = QHBoxLayout()
        recon_mode_layout.addWidget(basic_button)
        recon_mode_layout.addWidget(hilo_button)
        self.hilo_bool = False
        hilo_button.clicked.connect(lambda: self.toggle_hilo(True))
        basic_button.clicked.connect(lambda: self.toggle_hilo(False))
        lfm_param_layout.addLayout(recon_mode_layout)

        # Layouts horizontaux pour chaque bouton et son champ de texte
        zstack_layout = QHBoxLayout()
        perspectives_layout = QHBoxLayout()
        # Bouton et champ de texte pour "Z Stack"
        zstack_button = QPushButton("Save")
        zstack_button.adjustSize()
        zstack_label = QLabel("Z Stack:")
        self.zstack_name_input = QLineEdit(self)
        self.zstack_name_input.setPlaceholderText("Enter output path z stack")
        zstack_layout.addWidget(zstack_label)
        zstack_layout.addWidget(self.zstack_name_input)
        zstack_layout.addWidget(zstack_button)

        # Bouton et champ de texte pour "Perspectives"
        perspectives_button = QPushButton("Save")
        perspectives_button.adjustSize()
        perspectives_label = QLabel("Perspectives:")
        self.perspectives_name_input = QLineEdit(self)
        self.perspectives_name_input.setPlaceholderText("Enter output path perspectives")
        perspectives_layout.addWidget(perspectives_label)
        perspectives_layout.addWidget(self.perspectives_name_input)
        perspectives_layout.addWidget(perspectives_button)

        # Ajout des layouts au layout principal horizontal
        buttons_layout = QVBoxLayout()
        buttons_layout.addLayout(perspectives_layout)
        buttons_layout.addLayout(zstack_layout)
        lfm_param_layout.addLayout(buttons_layout)
        lfm_param_group.setLayout(lfm_param_layout)

        # Connect buttons to methods from self.lfm
        zstack_button.clicked.connect(lambda: self.run_long_task(zstack_button, self.zstack_name_input, self.left_layout ,self.lfm.reconstruction_zstack, self.hilo_bool, self.file_paths[0], self.file_paths[1], self.file_paths[2], self.zstack_name_input.text(), self.file_paths[4]))
        perspectives_button.clicked.connect(lambda: self.run_long_task(perspectives_button, self.perspectives_name_input, self.right_layout, self.lfm.reconstruction_perspectives, self.hilo_bool, self.file_paths[0], self.file_paths[1], self.file_paths[2], self.perspectives_name_input.text(), self.file_paths[4]))
        
        

        # Add widgets groups to the grid layout

        layout.addWidget(cameraparam_group, 0, 0)
        layout.addWidget(laser_group, 1, 0)
        layout.addWidget(dmd_group, 2, 0)
        layout.addWidget(lfm_param_group, 3, 0)
        
        layout.addWidget(camera_group, 0, 1, 3, 1)
        layout.addWidget(lfm_group, 3, 1, 1, 1)

        layout.setColumnStretch(0, 1)  # Colonne de gauche (1/3)
        layout.setColumnStretch(1, 2)  # Colonne de droite (2/3)
        
        






    #################### Camera ####################

    def toggle_video(self, button):
        """Start or stop video capture."""
        if button.isChecked():
            self.video_worker = CAMWorker(self.camera)
            self.video_worker.signals.frame_ready.connect(self.update_image)
            # self.video_worker_2.signals.camera_error.connect(self.handle_error)
            # self.video_worker_2.signals.camera_info.connect(self.update_camera_info)
            self.threadpool.start(self.video_worker)
            button.setText("Stop Live")
        else:
            self.video_worker.stop()
            button.setText("Live")

    def take_snapshot(self, path=None, idx=None):
        """Capture a single frame."""
        # if self.video_worker._running:
        #     self.video_worker.take_snap()
        #     print("Image_captured")
        # else:
        #     print("Start Live preview first")
        self.video_worker.take_snap(path)
        if path:
            self.file_paths[idx] = path # Store the actual file path in the list
            self.video_worker.take_snap(path)
        else:
            self.video_worker.take_snap()
        print("Image_captured")

    def set_expo(self, exp_time):
        """Change l'exposition sans freeze l'affichage."""
        if self.video_worker and hasattr(self.video_worker, 'set_exposure'):
            running = self.video_worker._running  # Vérifie si l'acquisition est active
            self.video_worker.set_exposure(exp_time)  # Applique la nouvelle valeur
            print(f"Exposure set to {self.video_worker.exposure} s")

    def set_gain(self, gain):
        if self.video_worker and hasattr(self.video_worker, 'set_gain'):
            self.video_worker.set_gain(gain)
            print(f"Changed gain to {int(gain)}")

    def update_image(self, frame):
        """Update the display with new frame."""
        self.image_view.setImage(frame, autoRange=False, autoLevels=False,
                                autoHistogramRange=False)
    
    def handle_cursor_position(self, pos, value):
        """Handle updates to the cursor position and intensity."""
        # print(f"Cursor at x={pos.x()}, y={pos.y()} with intensity value {value:.2f}")
        self.pos_info_label.setText(f"X:{pos.x()} | Y:{pos.y()} | Intensity:{value:.2f}")

##############################################################################

    def capture_snapshot(self):
        """ Save snapshot from the camera """
        frame = self.camera.snap()
        if frame is not None:
            filename, _ = QFileDialog.getSaveFileName(self, "Save Snapshot", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
            if filename:
                qimg = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1], QImage.Format.Format_Grayscale8)
                qimg.save(filename)


    def update_frame(self, frame):
        """Convert image in displayable format for Qt."""
        height, width = frame.shape
        bytes_per_line = width
        qimage = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        self.camera_view.setPixmap(pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio))

    def update_exposure_from_slider(self):
        """Update exposition and synchronizes slider and qlineedit text box."""
        value = self.exposure_slider.value()
        self.exposure_input.setText(str(value))  # Met à jour le champ texte
        self.set_expo(value)  # Envoie la valeur à la caméra

    def update_exposure_from_input(self):
        """Update exposition and synchronizes slider and qlineedit text box."""
        try:
            value = int(self.exposure_input.text())  # Convertit l'entrée en int
            if 1 <= value <= 1000:  # Vérifie si dans la plage valide
                self.exposure_slider.setValue(value)  # Met à jour le slider
                self.set_expo(value)  # Envoie à la caméra
            else:
                print("Value out of range")
        except ValueError:
            print("Invalid entry")

    def update_axial_from_slider(self):
            """Update axial position of focus with thorlabs motor."""
            value = self.axial_slider.value()
            self.axial_input.setText(str(value))
            self.motorx.move_to(value*1e-3)
            print(f"Axial translation to {value} um")
    
    def update_axial_from_input(self):
        """Update axial position and synchronizes slider and qlineedit text box."""
        try:
            value = int(self.axial_input.text())  # Convertit l'entrée en int
            if 4000 <= value <= 5800:  # Vérifie si dans la plage valide
                self.axial_slider.setValue(value)  # Met à jour le slider
                self.motorx.move_to(value*1e-3)  # Envoie au moteur thorlabs
                print(f"Axial translation to {value} um")
            else:
                print("Value out of range")
        except ValueError:
            print("Invalid entry")


    ################# Laser ###################

    def toggle_button_laser(self, button, laser):
        if button.isChecked():
            button.setStyleSheet("background-color: lightgreen;")
            button.setText("ON")
            laser.turn_on()
        else:
            button.setStyleSheet("background-color: lightgray;")
            button.setText("OFF")
            laser.turn_off()
    
    def update_power_from_slider(self, value):
        """Update exposition and synchronizes slider and qlineedit text box."""
        self.power_input.setText(str(value))  # Met à jour le champ texte
        self.laser.set_power(value)  # Envoie value au laser

    def update_power_from_input(self):
        """Update exposition and synchronizes slider and qlineedit text box."""
        try:
            value = int(self.power_input.text())  # Convertit l'entrée en int
            if 1 <= value <= 1000:  # Vérifie si dans la plage valide
                self.power_slider.setValue(value)  # Met à jour le slider
                self.laser.set_power(value)  # Envoie value au laser
            else:
                print("Value out of range")
        except ValueError:
            print("Invalid entry")


    ############## DMD ##################
    
    def toggle_button_dmd(self, button, dlp):
        if button.isChecked():
            button.setStyleSheet("background-color: lightgreen;")
            button.setText("ON")
            dlp.exitStandby()
        else:
            button.setStyleSheet("background-color: lightgray;")
            button.setText("OFF")
            dlp.enterStandby()

    
    ######### FLM Reconstruction and visualisation #######

    def browse_file(self, line_edit):
        """Opens a QFileDialog to select an stack."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*)")
        if file_path:
            line_edit.setText(file_path)
            
    def browse_image(self, file_input, index):
        """Opens a QFileDialog to select an image and updates the input field."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.tiff *.tif)")
        if file_path:
            file_input.setText(file_path)
            self.file_paths[index] = file_path # Store the actual file path in the list
            print(self.file_paths)
    
    def browse_dir(self, file_input, index):
        """Opens a QFileDialog to select a directory and updates the input field."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory", "")
        if dir_path:
            file_input.setText(dir_path)
            self.file_paths[index] = dir_path  # Store the actual directory path in the list
            print(self.file_paths)

    def save_file(self, file_input):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Images (*.tiff *.tif)")
        if file_path:
            file_input.setText(file_path)
    
    def save_snap(self, file_input, index):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Images (*.tiff *.tif)")
        if file_path:
            file_input.setText(file_path)
            self.file_paths[index] = file_path
            self.take_snapshot(file_path, index)

    def visualize_image(self, layout, qline_edit):
        """Displays the chosen interactive image."""
        image_path = qline_edit.text()
        self.viewer = ImageStackViewer()
        self.viewer.load_stack(image_path)
        qline_edit.setText(image_path)    # Met à jour le QLineEdit avec le nouveau chemin de fichier
        widget = layout.itemAt(0).widget()  # Find label in layout
        if widget:
            layout.removeWidget(widget)
            widget.deleteLater() # Delete label
        layout.insertWidget(0, self.viewer.get_widget())  # Add widget (image) at label place
    
    def handle_input_source_change(self, text):
        """Handle input source selector for the DMD."""
        if text == "Flash":
            self.dlp.setInputSource(source=2)
        elif text == "Internal Test Patterns":
            self.dlp.setInputSource(source=1)

    def toggle_hilo(self, is_hilo):
        """Active ou désactive le mode HiLo"""
        self.hilo_bool = is_hilo
        if is_hilo:
            print("HiLo recontruction mode.")
            # Tu peux aussi mettre à jour les styles ici
        else:
            print("Basic recontruction mode.")


    def run_long_task(self, button: QPushButton, file_input, layout, method, *args):
        """Exécute une méthode longue en mettant à jour le bouton."""
        
        self.save_file(file_input)
        args = list(args)
        args[4] = file_input.text()
        
        initial_text = button.text()
        button.setText("Running..")  # Change le texte du bouton
        button.setEnabled(False)  # Désactive le bouton

        # Création du thread et du worker
        self.worker_thread = QThread()
        self.worker = LFMWorker(method, *args)
        self.worker.moveToThread(self.worker_thread)

        # Connexion des signaux
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(lambda text: button.setText(text))
        self.worker.finished.connect(lambda: button.setText(initial_text))
        self.worker.finished.connect(lambda: button.setEnabled(True))
        if layout == self.left_layout:
            self.worker.finished.connect(lambda: self.visualize_image(layout, file_input))
        if layout == self.right_layout:
            self.worker.finished.connect(lambda: self.visualize_image(layout, file_input))
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        # Démarrer le thread
        self.threads.append(self.worker_thread)
        self.worker_thread.start()
    