import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# --- 1. Charger les stacks ---
lfm_stack = tifffile.imread("E:/surgele/juillet/tests_2107/fourierslice60_padded.tif")
rl_stack = tifffile.imread("E:/surgele/juillet/tests_2107/fourierslice60_padded_RL20.tif")
hilo_stack = tifffile.imread("E:/surgele/juillet/tests_2107/hilo2_fourierslice60_padded_sigma08.tiff")

# Vérif dimensions (Z, Y, X)
print(lfm_stack.shape)

# --- 2. Sélectionner la ROI à la souris sur la première image ---
fig, ax = plt.subplots()
ax.imshow(lfm_stack[0], cmap="gray")
plt.title("Cliquez pour centre de la ROI")
pts = plt.ginput(1)  # clic souris
plt.close(fig)

cy, cx = int(pts[0][1]), int(pts[0][0])
r_signal = 30   # rayon ROI signal en pixels (à ajuster selon taille capsule)
r_bg_in = r_signal + 3  # bord interne du fond
r_bg_out = r_signal + 8 # bord externe du fond

print(f"Centre ROI = ({cx}, {cy}), rayon = {r_signal}")

# --- 3. Créer des masques (signal + background) ---
def make_circular_mask(shape, center, r_in, r_out=None):
    Y, X = np.ogrid[:shape[0], :shape[1]]
    dist = (X - center[0])**2 + (Y - center[1])**2
    if r_out is None:  # disque plein
        return dist <= r_in**2
    else:  # anneau
        return (dist >= r_in**2) & (dist <= r_out**2)

mask_signal = make_circular_mask(lfm_stack[0].shape, (cx, cy), r_signal)
mask_bg = make_circular_mask(lfm_stack[0].shape, (cx, cy), r_bg_in, r_bg_out)

# --- 4. Fonction pour extraire profils ---
def extract_profile(stack, mask_signal, mask_bg):
    signal = stack[:, mask_signal].mean(axis=1)
    background = stack[:, mask_bg].mean(axis=1)
    sbr = signal / (background + 1e-9)  # éviter div/0
    return signal, background, sbr

sig_lfm, bg_lfm, sbr_lfm = extract_profile(lfm_stack, mask_signal, mask_bg)
sig_rl, bg_rl, sbr_rl = extract_profile(rl_stack, mask_signal, mask_bg)
sig_hilo, bg_hilo, sbr_hilo = extract_profile(hilo_stack, mask_signal, mask_bg)

# --- 5. Tracer profils SBR comparatifs ---
n_planes = len(sig_lfm)
z = np.linspace(-60, 60, n_planes)  # profondeur en µm

plt.figure(figsize=(7,5))
plt.plot(z, sbr_lfm, label="LFM classique")
plt.plot(z, sbr_rl, label="LFM RL (20 it)")
plt.plot(z, sbr_hilo, label="HiLo-LFM")
plt.xlabel("Profondeur axiale z (µm)")
plt.ylabel("SBR (signal/fond)")
plt.legend()
plt.title("Comparaison du sectionnement optique")
plt.show()


# --- 6) Afficher la slice à z = 0 µm (LFM classique) avec ROI + anneau de fond ---
idx_0um = int(np.argmin(np.abs(z - 0.0)))  # index du plan le plus proche de 0 µm
slice_0um = lfm_stack[idx_0um]

fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(slice_0um, cmap="gray")

# ROI signal (rouge)
circ_signal = Circle((cx, cy), r_signal, fill=False, linewidth=2, edgecolor="red", label="ROI signal")
ax.add_patch(circ_signal)

# Anneau de fond (vert) : deux cercles (interne + externe)
circ_bg_in = Circle((cx, cy), r_bg_in, fill=False, linewidth=1.5, linestyle="--", edgecolor="green", label="Fond (anneau)")
circ_bg_out = Circle((cx, cy), r_bg_out, fill=False, linewidth=1.5, linestyle="--", edgecolor="green")
ax.add_patch(circ_bg_in)
ax.add_patch(circ_bg_out)

ax.set_title(f"Slice LFM classique — z ≈ {z[idx_0um]:.1f} µm")
ax.legend(loc="upper right", frameon=True)
ax.set_axis_off()
plt.show()

