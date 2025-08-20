from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSlider
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
import tifffile as tiff
import textwrap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class ImageStackViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.stack = None
        self.stack_type = None
        self.row_idx, self.col_idx = 7, 7  # Pour stacks perspectives
        self.z_idx = 20  # Pour stacks verticaux

        # Création de la figure Matplotlib intégrée
        self.fig, self.ax = plt.subplots(figsize=(3, 3))
        self.canvas = FigureCanvas(self.fig)

        # Slider pour naviguer dans le stack
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.update_index)

        # Layout vertical pour placer l'image + slider
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.slider)
        self.setLayout(self.layout)

    def load_stack(self, im_path):
        """Charge l'image TIFF et détecte le type de stack."""
        self.stack = tiff.imread(im_path)

        if self.stack.shape == (225, 140, 140):  # Stack perspectives (15x15)
            self.stack_type = 'perspective'
            self.stack = self.stack.reshape(15, 15, 140, 140)
            self.slider.setMaximum(14)  # Navigation entre 0 et 14 pour les perspectives
            self.slider.setValue(self.row_idx)
        else:  # Stack vertical
            self.stack_type = 'vertical'
            self.slider.setMaximum(self.stack.shape[0] - 1)
            self.slider.setValue(self.z_idx)

        self.display_slice()  # Affiche la première image

    def wrap_title(self, title, max_width=30):
        """Gère les titres trop longs."""
        return "\n".join(textwrap.wrap(title, width=max_width))

    def display_slice(self):
        """Affiche une image de la stack en fonction de l'index courant."""
        self.ax.clear()

        if self.stack_type == 'perspective':
            img = self.stack[self.row_idx, self.col_idx]
            title = f'R{self.row_idx} - C{self.col_idx}'
        else:
            img = self.stack[self.z_idx]
            title = f'Z {self.z_idx}'

        self.ax.imshow(img, cmap='gray')
        self.ax.set_title(self.wrap_title(title))
        self.ax.axis('off')
        self.canvas.draw()  # Met à jour l'affichage

    def update_index(self, value):
        """Mise à jour de l'affichage quand le slider change de valeur."""
        if self.stack_type == 'perspective':
            self.row_idx = value
            self.col_idx = value
        else:
            self.z_idx = value
        self.display_slice()  # Rafraîchit l'affichage

    def get_widget(self):
        """Retourne le QWidget contenant l'affichage et le slider."""
        return self
