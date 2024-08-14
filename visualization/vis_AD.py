import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
from skimage import exposure
from skimage.transform import resize

name = 'Patient_128_2/'



f = plt.figure(figsize=(14, 6), dpi=500)  # Reduced figure width
model_paths = ['/home/amishr17/aryan/new_attempt/results/vis/upcol/' + 'TBAD-patient2_UPCoL']
# model_paths = ['/home/amishr17/aryan/new_attempt/results/vis/upcol/' + 'TBAD-91_UPCoL']

labels = ["False Lume(FL)", "True Lumen(TL)"]
cp = ["red", "yellow"]
patches = [mpatches.Patch(color=cp[i], label=labels[i]) for i in range(len(cp))]
cmap = matplotlib.colors.ListedColormap(["gray", "red", "yellow"])
grid = plt.GridSpec(3, 3, wspace=0.05, hspace=0.05)  

plt.legend(handles=patches, ncol=4, fontsize=12, loc="upper center", bbox_to_anchor=(0.5, 1.05))
# ax = 1

# img = np.load('/home/amishr17/aryan/new_attempt/numpy_preprocess/' +  'Image.npy', allow_pickle=True).squeeze()
# label = np.load('/home/amishr17/aryan/new_attempt/numpy_pre
# /home/amishr17/aryan/new_attempt/numpy_preprocess/Image.npy
# /home/amishr17/aryan/new_attempt/results/vis/ilnpy/TBAD-patient3-Image.npy
img = np.load('/home/amishr17/aryan/new_attempt/results/vis/ilnpy/' +  'TBAD-patient2-Image.npy', allow_pickle=True).squeeze()
# img= resize(i, (64, 64, 64), anti_aliasing=True)
label = np.load('/home/amishr17/aryan/new_attempt/results/vis/ilnpy/' + 'TBAD-patient2-Label.npy', allow_pickle=True).squeeze()
# img = np.load('/home/amishr17/aryan/new_attempt/numpy_preprocess/' +  'Image.npy', allow_pickle=True).squeeze()
# label = np.load('/home/amishr17/aryan/new_attempt/numpy_preprocess/' + 'Label.npy', allow_pickle=True).squeeze()

assert img.ndim == 3, "Image data should be 3-dimensional"
assert label.ndim == 3, "Label data should be 3-dimensional"

z_size, y_size, x_size = img.shape

slice_ind_1 = 68# x-axis:   Sagittal 
slice_ind_2 = 79 # y-axis:  Coronal
slice_ind_3 = 65  # z-axis: Axial

img_alpha = 0.99
label_alpha = 0.6

def enhance_image(image):
    return exposure.adjust_gamma(exposure.equalize_hist(image), 0.8)

def add_crosshairs(ax, x, y, color='white'):
    ax.axhline(y=y, color=color, linestyle='--', linewidth=0.5)
    ax.axvline(x=x, color=color, linestyle='--', linewidth=0.5)
    ax.text(0.02, 0.98, f'({x}, {y})', color=color, transform=ax.transAxes,
            verticalalignment='top', fontsize=8)


plt.xticks([])
plt.yticks([])
plt.axis('off')

# ... (rest of the setup code remains the same)

def add_info_textbox(fig, model_name, dice_score, input_image, output_image, input_dim, output_dim):
    text = f"Model Name: {model_name}\n\n"
    text += f"Dice Score: {dice_score}\n"
    text += f"Input Image:{input_image}\n"
    text += f"Input Dimensions:{input_dim}\n"
    text += f"Output Image:{output_image}\n"
    text += f"Output Dimensions:{output_dim}\n\n"
    text += f"Class: 1: dice 0.0% | jaccard 0.0% | hd95 0.0 | asd 0.0\n"
    text+=  f"Class: 2: dice 0.0% | jaccard 0.0% | hd95 0.0 | asd 0.0"


    
    ax = fig.add_subplot(grid[:,1])  # Place textbox in the middle column
    ax.axis('off')
    ax.text(0.05, 0.5, text, fontsize=11, ha='left', va='center', bbox=dict(facecolor='white', alpha=0.8, pad=5))

for model_path in model_paths:
    predict = np.load(model_path + '.npy', allow_pickle=True).squeeze()
    print(predict.shape)
    assert predict.ndim == 3, f"Prediction data from {model_path} should be 3-dimensional"
   
    # Axial view (x-y plane)
    pred_3 = predict[:, :, slice_ind_3]
    ax3 = f.add_subplot(grid[0, 0])
    plt.xticks([])
    plt.yticks([])
    model = model_path.split('/')[-1]
    enhanced_img = enhance_image(img[:, :, slice_ind_3])
    plt.imshow((enhanced_img), alpha=img_alpha, cmap='gray')
    plt.imshow((pred_3.astype(np.uint8)), alpha=label_alpha, vmin=0, vmax=len(cmap.colors), cmap=cmap)
    add_crosshairs(ax3, slice_ind_2, slice_ind_1)

    # Coronal view (x-z plane)
    pred_2 = predict[:, slice_ind_2, :]
    ax2 = f.add_subplot(grid[1, 0])
    plt.xticks([])
    plt.yticks([])
    enhanced_img = enhance_image(np.rot90(img[:, slice_ind_2, :]))
    plt.imshow((enhanced_img), alpha=img_alpha, cmap='gray')
    plt.imshow((np.rot90(pred_2.astype(np.uint8))), alpha=label_alpha, vmin=0, vmax=len(cmap.colors), cmap=cmap)
    add_crosshairs(ax2, slice_ind_1, slice_ind_3)

    # Sagittal view (y-z plane)
    pred_1 = predict[slice_ind_1, :, :]
    ax1 = f.add_subplot(grid[2, 0])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(model, fontsize=10)
    enhanced_img = enhance_image(np.rot90(img[slice_ind_1, :, :]))
    plt.imshow((enhanced_img), alpha=img_alpha, cmap='gray')
    plt.imshow((np.rot90(pred_1.astype(np.uint8))), alpha=label_alpha, vmin=0, vmax=len(cmap.colors), cmap=cmap)
    add_crosshairs(ax1, slice_ind_2, slice_ind_3)

# Plot ground truth
# Axial view (x-y plane)
ax3 = f.add_subplot(grid[0, 2])
plt.xticks([])
plt.yticks([])
enhanced_img = enhance_image(img[:, :, slice_ind_3])
plt.imshow((enhanced_img), alpha=img_alpha, cmap='gray')
plt.imshow((label[:, :, slice_ind_3]), alpha=label_alpha, vmin=0, vmax=len(cmap.colors), cmap=cmap)
add_crosshairs(ax3, slice_ind_2, slice_ind_1)

# Coronal view (x-z plane)
ax2 = f.add_subplot(grid[1, 2])
plt.xticks([])
plt.yticks([])
enhanced_img = enhance_image(np.rot90(img[:, slice_ind_2, :]))
plt.imshow((enhanced_img), alpha=img_alpha, cmap='gray')
plt.imshow((np.rot90(label[:, slice_ind_2, :])), alpha=label_alpha, vmin=0, vmax=len(cmap.colors), cmap=cmap)
add_crosshairs(ax2, slice_ind_1, slice_ind_3)

# Sagittal view (y-z plane)
ax1 = f.add_subplot(grid[2, 2])
plt.xticks([])
plt.yticks([])
plt.xlabel('Ground Truth', fontsize=10)
enhanced_img = enhance_image(np.rot90(img[slice_ind_1, :, :]))
plt.imshow((enhanced_img), alpha=img_alpha, cmap='gray')
plt.imshow((np.rot90(label[slice_ind_1, :, :])), alpha=label_alpha, vmin=0, vmax=len(cmap.colors), cmap=cmap)
add_crosshairs(ax1, slice_ind_2, slice_ind_3)


model_name = "UPCoL"
dice_score = "0.916"  
input_image = "TBAD_patient2-Image.npy"
output_image = "TBAD_patient2_UPCoL.npy"
input_dim = f"{img.shape}"
output_dim = f"{predict.shape}"

add_info_textbox(f, model_name, dice_score, input_image, output_image, input_dim, output_dim)

plt.tight_layout(pad=0.5)  

output_dir = '/home/amishr17/aryan/new_attempt/results/vis/'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, name[:-1] + '_enhanced_with_info.png')

plt.savefig(output_path, bbox_inches='tight', dpi=f.dpi, pad_inches=0.05)  
