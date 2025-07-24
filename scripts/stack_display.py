import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import textwrap
from matplotlib.widgets import Slider



image_paths = [
    # "D:/surgele/new_billes/zstack3.tif",
    # "D:/surgele/new_billes/RL_zstack3_10iter.tif",
    # "D:/surgele/new_billes/RL_zstack3_10iter.tif",
    # "D:/surgele/new_billes/RL_zstack3_10iter.tif",
    # "D:/surgele/juin/tests_0406/new_hilo_zstack_fourierslice2.tiff",
    # "D:/surgele/juin/tests_0406/new_hilo_zstack_fourierslice2_padded.tiff",
    "D:/surgele/refocusing/billes2D_10um_step/persp/+10_perspectives.tiff"
]



row_idx, col_idx = 7, 7
z_idx = 20

stacks = [tiff.imread(path) for path in image_paths]

if stacks[0].shape == (225, 140, 140):
    stack_type = 'perspective'
    stacks = [stack.reshape(15, 15, 140, 140) for stack in stacks]
else:
    stack_type = 'vertical'


fig, axes = plt.subplots(1, len(stacks), figsize=(5 * len(stacks), 5))

if len(stacks) == 1:
    axes = [axes]



################## Functions #####################

def wrap_title(title, max_width=30):
    return "\n".join(textwrap.wrap(title, width=max_width))

def update_images():
    for img_display, ax, stack, path in zip(img_displays, axes, stacks, image_paths):
        if stack_type == 'perspective':
            img_display.set_data(stack[row_idx, col_idx])
            ax.set_title(wrap_title(f'{path} - R{row_idx} C{col_idx}'))
        else:
            img_display.set_data(stack[z_idx])
            ax.set_title(wrap_title(f'{path} - Z{z_idx}'))
    fig.canvas.draw_idle()

def on_key_event(event):
    global row_idx, col_idx, z_idx
    if stack_type == 'perspective':
        if event.key == 'right' and col_idx < 14:
            col_idx += 1
        elif event.key == 'left' and col_idx > 0:
            col_idx -= 1
        elif event.key == 'down' and row_idx < 14:
            row_idx += 1
        elif event.key == 'up' and row_idx > 0:
            row_idx -= 1
    else:
        if event.key == 'up' and z_idx < stacks[0].shape[0] - 1:
            z_idx += 1
        elif event.key == 'down' and z_idx > 0:
            z_idx -= 1

    update_images()

####################################################

img_displays = []
for ax, stack, path in zip(axes, stacks, image_paths):
    if stack_type == 'perspective':
        img = ax.imshow(stack[row_idx, col_idx], cmap='gray')
        ax.set_title(wrap_title(f'{path} - R{row_idx} C{col_idx}'))
    else:
        img = ax.imshow(stack[z_idx], cmap='gray')
        ax.set_title(wrap_title(f'{path} - Z{z_idx}'))
    ax.axis('off')
    img_displays.append(img)


fig.canvas.mpl_connect('key_press_event', on_key_event)
plt.show()
