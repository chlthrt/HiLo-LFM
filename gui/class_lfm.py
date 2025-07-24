import numpy as np
import os
import glob
import skimage.io
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
from scipy.ndimage import shift
from skimage.measure import regionprops, label, find_contours
from skimage.segmentation import flood
from tifffile import imwrite
import cv2
from circle_fit import taubinSVD
import new_hilo1
import fourierslice_refocusing
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtCore import QThread, pyqtSignal, QObject
import time  # Pour simuler un long traitement

class LFM:
    def __init__(self, mm_dir, save_dir, Nx=140, Ny=140):
        self.Nx = Nx
        self.Ny = Ny
        self.mm_dir = mm_dir
        self.save_dir = save_dir

    def pre_processing(self, file_path, bg_path=None, angle=-0.001, pitch=14.02):
        LFM1 = skimage.io.imread(file_path)
        if bg_path:
            bg = skimage.io.imread(bg_path)
            LFM1 = (LFM1 - bg)
        img_rotated = ndimage.rotate(LFM1, angle, reshape=False)  
        LFM1 = img_rotated.astype(np.int32)
        h, w = LFM1.shape[:2]
        fraction1 = 15/pitch
        fraction2 = 15/pitch
        img_resized = np.array(Image.fromarray(LFM1, mode='I').resize((int(fraction1 * w), int(fraction2 * h)), Image.Resampling.BILINEAR))
        return img_resized

    def calibrate_LFM3(self, img, x=16, y=20):
        step = 15
        LFM = np.zeros((self.Nx, self.Ny, step, step))
        XX = np.zeros((self.Nx, self.Ny))
        YY = np.zeros((self.Nx, self.Ny))
        CenterX = np.zeros((self.Nx, self.Ny))
        CenterY = np.zeros((self.Nx, self.Ny))
        CenterX_ini = np.zeros((self.Nx, self.Ny))
        CenterY_ini = np.zeros((self.Nx, self.Ny))
            
        for ii in range(self.Nx):
            for jj in range(self.Ny):

                if ii == 0 and jj == 0:
                    cx, cy = x, y
                elif ii == 0:
                    # première ligne : moyenne impossible, on prend celle de gauche
                    cx = CenterX[ii, jj - 1] + step
                    cy = CenterY[ii, jj - 1]
                elif jj == 0:
                    # premiere colonne : moyenne impossible, se base sur la lentille au-dessus
                    cx = CenterX[ii - 1, jj]
                    cy = CenterY[ii - 1, jj] + step
                else:
                    # moyenne entre la lentille de gauche et celle du dessus
                    cx = (CenterX[ii - 1, jj] + CenterX[ii, jj - 1] + step) / 2
                    cy = (CenterY[ii - 1, jj] + CenterY[ii, jj - 1] + step) / 2
                    
                cx = int(cx)
                cy = int(cy)

                # extraction de la microlentille dans l'image
                intermediate = img[cy-(step//2):cy+(step//2 + 1), cx-(step//2):cx+(step//2 + 1)]
                intermediate = intermediate.astype(float)
                
                max_pos = np.unravel_index(np.argmax(intermediate), intermediate.shape)
                thresh = np.percentile(intermediate, 30)
                flooded = flood(intermediate, seed_point=max_pos, tolerance=intermediate[max_pos]-thresh)

                ##### Contours + Circle-fit #####
                flooded = flooded.astype(np.uint8)
                contours, _ = cv2.findContours(flooded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                main_contour = min(contours, key=len)
                points = main_contour[:, 0, :]  # de (n,1,2) vers (n,2)
                fit = taubinSVD(points)
                ymax = round(fit[1])
                xmax = round(fit[0])

                dx = xmax - (step // 2)
                dy = ymax - (step // 2)

                XX[ii, jj] = dx
                YY[ii, jj] = dy

                # ajustement du centre
                cx_adj = cx + dx
                cy_adj = cy + dy
                CenterX_ini[ii, jj] = cx
                CenterY_ini[ii, jj] = cy
                CenterX[ii, jj] = cx_adj
                CenterY[ii, jj] = cy_adj

                # Ré-extraction de la micro-image ajustée
                LFM[ii, jj] = img[cy_adj-(step//2):cy_adj+(step//2 + 1), cx_adj-(step//2):cx_adj+(step//2 + 1)]

        return LFM, CenterX, CenterY, XX, YY
    
    def compute_intensity(self, LFM):
        I0 = np.sum(LFM, axis=(2, 3))
        return I0
    
    def compute_LFM(self, img, CenterX, CenterY, CX, CY):
        step = 15
        LFM = np.zeros([self.Nx, self.Ny, 15, 15])
        for ii in range(0, self.Nx):
            for jj in range(0, self.Ny):    
                LFM[ii, jj, :, :] = img[int(CenterY[ii, jj])-(step//2):int(CenterY[ii, jj])+(step//2 + 1), 
                                        int(CenterX[ii, jj])-(step//2):int(CenterX[ii, jj])+(step//2 + 1)]   
        return LFM
    
    def rendered_focus(self, C, rendered_height, rendered_width, side, radiance, returnVal=False):
        rendered = np.zeros((rendered_height, rendered_width))
        center_uv = int(side / 2)
        for u in range(side):
            for v in range(side):
                shift_x, shift_y = C * (center_uv - u), C * (center_uv - v)
                rendered[:, :] += shift(radiance[:, :, u, v], (shift_x, shift_y))
        final = rendered / (side * side)
        if returnVal:
            return final
        
    def reconstruction_perspectives(self, hilo_bool, file_path0, bg_path, file_path, output_file_path, dir_path_patterns=None):
        
        calibration_image=self.pre_processing(file_path0)
        bright=self.pre_processing(file_path, bg_path)

        # Uniform illumination
        LFM,centerX,centerY,CX,CY=self.calibrate_LFM3(calibration_image)
        LFM01=self.compute_LFM(bright,centerX,centerY, CX, CY)                               
        LFM02=LFM01
        aa=self.compute_intensity(LFM01)
        rendered_height,rendered_width=LFM01.shape[0],LFM01.shape[1]
        side=LFM02.shape[2]                                          
        radiance=LFM01    

        # Structured illumination
        if hilo_bool==True:
            file_path_patternS = sorted(glob.glob(os.path.join(dir_path_patterns,'*.tiff')))
            radiance_patternS = []
            side_patternS = []
            rendered_height_patternS = []
            rendered_width_patternS = []
            for file_path_pattern in file_path_patternS:
                pattern = self.pre_processing(file_path_pattern, bg_path)
                LFM01_pattern = self.compute_LFM((pattern/(calibration_image+1))*np.mean(calibration_image),centerX,centerY,CX,CY)
                rendered_height_pattern,rendered_width_pattern=LFM01_pattern.shape[0],LFM01_pattern.shape[1]
                side_pattern=LFM01_pattern.shape[2]                                          
                radiance_pattern=LFM01_pattern
                radiance_patternS.append(radiance_pattern)
                side_patternS.append(side_pattern)
                rendered_height_patternS.append(rendered_height_pattern)
                rendered_width_patternS.append(rendered_width_pattern)
            
            hilo_perspectives = []
            for i in range(radiance.shape[3]):
                for j in range(radiance.shape[3]):
                    radiance_structured = [r[:, :, i, j] for r in radiance_patternS]
                    hilo_perspective, _, _, _ = new_hilo1.basic_hilo(radiance[:, :, i, j], radiance_structured)
                    hilo_perspectives.append(hilo_perspective)
            hilo_perspectives_stack = np.stack(hilo_perspectives)

            imwrite(output_file_path, hilo_perspectives_stack)
            print("HiLo perspectives reconstruction saved!")
            return hilo_perspectives_stack

        elif hilo_bool == False:
            perspectives_stack = np.stack([radiance[:, :, i, j] for i in range(radiance.shape[3]) for j in range(radiance.shape[3])])
            imwrite(output_file_path, perspectives_stack)
            print("Classic perspectives reconstruction saved!")
            return perspectives_stack

    

    def reconstruction_zstack(self, hilo_bool, file_path0, bg_path, file_path, output_file_path, dir_path_patterns=None):
        
        #hilo reconstruction
        if hilo_bool == True:
            hilo_perspectives_stack = self.reconstruction_perspectives(hilo_bool, file_path0, bg_path, file_path, output_file_path, dir_path_patterns)
            
            U = V = int(np.sqrt(hilo_perspectives_stack.shape[0]))
            X, Y = hilo_perspectives_stack.shape[1], hilo_perspectives_stack.shape[2]
            radiance_hilo = hilo_perspectives_stack.reshape(U, V, X, Y).transpose(2, 3, 0, 1)

            hilo_refocus_stack, hilo_fft_stack, Z = fourierslice_refocusing.refocus_test(radiance_hilo, min=-150, max=150, Padding=False)
            imwrite(output_file_path, hilo_refocus_stack)
            print("HiLo Z stack reconstruction saved!")
        
        elif hilo_bool == False:
            perspectives_stack = self.reconstruction_perspectives(hilo_bool, file_path0, bg_path, file_path, output_file_path, dir_path_patterns)

            U = V = int(np.sqrt(perspectives_stack.shape[0]))
            X, Y = perspectives_stack.shape[1], perspectives_stack.shape[2]
            radiance = perspectives_stack.reshape(U, V, X, Y).transpose(2, 3, 0, 1)

            refocus_stack, fft_stack, Z = fourierslice_refocusing.refocus_test(radiance, min=-150, max=150, Padding=True)
            imwrite(output_file_path, refocus_stack)
            print("Classic Z stack reconstruction saved!")

    





class LFMWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(str)

    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def run(self):
        """Exécute la fonction longue et émet les signaux."""
        self.progress.emit("Running...")  # Met à jour le bouton
        self.function(*self.args, **self.kwargs)  # Appelle la fonction
        self.finished.emit()  # Signale la fin du traitement
