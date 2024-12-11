#%% Imports -------------------------------------------------------------------

import re
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Functions
from functions import open_czi

# Scipy
from scipy.ndimage import distance_transform_edt

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path('D:/local_Masschelein/data')

# Parameters
crop_size = 120

# Patient id
ctr = [1, 3, 4, 5, 6, 7, 9, 11]
pad = [2, 8, 10, 12, 13, 14, 17, 18, 19]

#%% Function(s) ---------------------------------------------------------------

def analyse(imgs, coords, crop_size):
    
    # Get edt data
    edt = np.zeros((crop_size, crop_size))
    edt[crop_size // 2, crop_size // 2] = 1
    edt = distance_transform_edt(1 - edt).astype(int)
    unique, counts = np.unique(edt, return_counts=True) 
    
    # Crop & analyse data 
    crops, avg_1Ds, avg_1Ds_max, avg_2Ds = [], [], [], [] 
    for i, img in enumerate(imgs):
        img = np.pad(img, pad_width=crop_size // 2)
        img_coords = coords[coords[:, 0] == i]
        if img_coords.shape[0] > 0:
            for coord in img_coords:    
                y, x = coord[1], coord[2]
                crop = img[y:y + crop_size, x:x + crop_size]
                avg_1D = np.bincount(edt.ravel(), weights=crop.ravel())
                avg_1D /= counts
                avg_1D /= avg_1D[0] # normalize to center point
                avg_1D_max = np.argmax(avg_1D)
                mapper = np.zeros(edt.max().astype(int) + 1)
                mapper[unique.astype(int)] = avg_1D
                avg_2D = mapper[edt.astype(int)]
                
                # Append
                crops.append(crop)
                avg_1Ds.append(avg_1D)
                avg_1Ds_max.append(avg_1D_max)
                avg_2Ds.append(avg_2D)
                     
    return crops, avg_1Ds, avg_1Ds_max, avg_2Ds


#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    t0 = time.time()
    
    # Paths 
    paths = list(data_path.glob("*.czi"))
    
    # Open data
    C1s, C2s = open_czi(paths)
    coords_path = Path(data_path, "coords.csv")
    coords = np.loadtxt(coords_path, delimiter=",").astype(int)
    
    # Get patient id
    patient_ids = []
    img_ids = coords[:, 0]
    for img_id in img_ids:
        match = re.match(r"(\d+)\.\d+\.\w+", paths[img_id].name)
        patient_ids.append(int(match.group(1)))
    
    # Analyse
    C1_crops, C1_avg_1Ds, C1_avg_1Ds_max, C1_avg_2Ds = analyse(
        C1s, coords, crop_size)
    C2_crops, C2_avg_1Ds, C2_avg_1Ds_max, C2_avg_2Ds = analyse(
        C2s, coords, crop_size)
    
    # Average data
    
    m_C1_avg_1Ds_ctr = [data for i, data in enumerate(C1_avg_1Ds) if patient_ids[i] in ctr]
    m_C1_avg_1Ds_pad = [data for i, data in enumerate(C1_avg_1Ds) if patient_ids[i] in pad]
    C1_avg_1Ds_ctr_avg = np.nanmean(np.stack(m_C1_avg_1Ds_ctr), axis=0)
    C1_avg_1Ds_pad_avg = np.nanmean(np.stack(m_C1_avg_1Ds_pad), axis=0)

    m_C2_avg_1Ds_ctr = [data for i, data in enumerate(C2_avg_1Ds) if patient_ids[i] in ctr]
    m_C2_avg_1Ds_pad = [data for i, data in enumerate(C2_avg_1Ds) if patient_ids[i] in pad]
    C2_avg_1Ds_ctr_avg = np.nanmean(np.stack(m_C2_avg_1Ds_ctr), axis=0)
    C2_avg_1Ds_pad_avg = np.nanmean(np.stack(m_C2_avg_1Ds_pad), axis=0)
    
    # ---
    
    m_C1_avg_1Ds_max_ctr = [data for i, data in enumerate(C1_avg_1Ds_max) if patient_ids[i] in ctr]
    m_C1_avg_1Ds_max_pad = [data for i, data in enumerate(C1_avg_1Ds_max) if patient_ids[i] in pad]

    m_C2_avg_1Ds_max_ctr = [data for i, data in enumerate(C2_avg_1Ds_max) if patient_ids[i] in ctr]
    m_C2_avg_1Ds_max_pad = [data for i, data in enumerate(C2_avg_1Ds_max) if patient_ids[i] in pad]
    
    # ---
    
    m_C1_crops_ctr = [data for i, data in enumerate(C1_crops) if patient_ids[i] in ctr]
    m_C1_crops_pad = [data for i, data in enumerate(C1_crops) if patient_ids[i] in pad]
    C1_crops_ctr_avg = np.nanmean(np.stack(m_C1_crops_ctr), axis=0)
    C1_crops_pad_avg = np.nanmean(np.stack(m_C1_crops_pad), axis=0)
    
    m_C2_crops_ctr = [data for i, data in enumerate(C2_crops) if patient_ids[i] in ctr]
    m_C2_crops_pad = [data for i, data in enumerate(C2_crops) if patient_ids[i] in pad]
    C2_crops_ctr_avg = np.nanmean(np.stack(m_C2_crops_ctr), axis=0)
    C2_crops_pad_avg = np.nanmean(np.stack(m_C2_crops_pad), axis=0)

    # ---
                
    t1 = time.time()
    print(f"runtime = {t1 - t0:.5f}")
         
    # # Display
    # import napari
    # viewer = napari.Viewer()
    # viewer.add_image(C1_crops_pad_avg)
    # viewer.add_image(C1_crops_ctr_avg)
    # viewer.add_image(C2_crops_pad_avg)
    # viewer.add_image(C2_crops_ctr_avg)
    
    # # Display
    # import napari
    # viewer = napari.Viewer()
    # viewer.add_image(np.stack(C1_crops))
    # viewer.add_image(np.stack(C2_crops))
    # viewer.add_image(np.stack(C1_avg_2Ds))
    # viewer.add_image(np.stack(C2_avg_2Ds))
    
#%% Plot

fig = plt.figure(figsize=(10, 10))
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# -----------------------------------------------------------------------------

# Plot 1
ax1.set_title("???")
ax1.plot(C1_avg_1Ds_ctr_avg)
ax1.plot(C1_avg_1Ds_pad_avg)
# ax1.set_ylim(-0.1, 1.1)
# ax1.set_xlim(-100, 2100)

# Plot 2
ax2.set_title("???")
ax2.boxplot(m_C1_avg_1Ds_max_ctr, positions=[1], widths=0.6)
ax2.boxplot(m_C1_avg_1Ds_max_pad, positions=[2], widths=0.6)

# Plot 3
ax3.set_title("???")
ax3.plot(C2_avg_1Ds_ctr_avg)
ax3.plot(C2_avg_1Ds_pad_avg)
# ax1.set_ylim(-0.1, 1.1)
# ax1.set_xlim(-100, 2100)

# Plot 4
ax4.set_title("???")
ax4.boxplot(m_C2_avg_1Ds_max_ctr, positions=[1], widths=0.6)
ax4.boxplot(m_C2_avg_1Ds_max_pad, positions=[2], widths=0.6)