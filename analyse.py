#%% Imports -------------------------------------------------------------------

import re
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Functions
from functions import open_czi

# Scipy
from scipy import stats
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

def analyse(imgs, coords, crop_size, cond1=ctr, cond2=pad):

    # Nested function(s) ------------------------------------------------------
    
    def average_data(crops, radInts, radInts_max, ids, cond):
        
        crops_cond, radInts_cond, radInts_max_cond = [], [], []
        for i in range(len(ids)):
            if ids[i] in cond:
                crops_cond.append(crops[i])
                radInts_cond.append(radInts[i])
                radInts_max_cond.append(radInts_max[i])
        crops_cond_avg = np.nanmean(np.stack(crops_cond), axis=0)
        radInts_cond_avg = np.nanmean(np.stack(radInts_cond), axis=0)
        radInts_cond_std = np.nanstd(np.stack(radInts_cond), axis=0)
        
        data_cond = {
            
            "crops_cond_avg" : crops_cond_avg,
            "radInts_cond_avg" : radInts_cond_avg,
            "radInts_cond_std" : radInts_cond_std,
            "radInts_max_cond" : radInts_max_cond,
            
            }
        
        return data_cond
          
    # Execute -----------------------------------------------------------------
    
    # Get ids
    ids = []
    img_ids = coords[:, 0]
    for img_id in img_ids:
        match = re.match(r"(\d+)\.\d+\.\w+", paths[img_id].name)
        ids.append(int(match.group(1)))
    
    # Get edt
    edt = np.zeros((crop_size, crop_size))
    edt[crop_size // 2, crop_size // 2] = 1
    edt = distance_transform_edt(1 - edt).astype(int)
    unique, counts = np.unique(edt, return_counts=True) 
    
    # Crop & analyse data 
    crops, radInts, radInts_max = [], [], [] 
    for i, img in enumerate(imgs):
        img = np.pad(img, pad_width=crop_size // 2)
        img_coords = coords[coords[:, 0] == i]
        if img_coords.shape[0] > 0:
            for coord in img_coords:    
                y, x = coord[1], coord[2]
                crop = img[y:y + crop_size, x:x + crop_size]
                radInt = np.bincount(edt.ravel(), weights=crop.ravel())
                radInt /= counts
                radInt /= radInt[0] # normalize to center point
                radInt_max = np.argmax(radInt)
                # mapper = np.zeros(edt.max().astype(int) + 1)
                # mapper[unique.astype(int)] = radInt
                # avg_2D = mapper[edt.astype(int)]
                
                # Append
                crops.append(crop)
                radInts.append(radInt)
                radInts_max.append(radInt_max)
                # avg_2Ds.append(avg_2D)
                     
    # Average data
    data_cond1 = average_data(crops, radInts, radInts_max, ids, cond1)
    data_cond2 = average_data(crops, radInts, radInts_max, ids, cond2)
    
    # Statistics
    radInts_max_t_stat, radInts_max_p_value = stats.ttest_ind(
        data_cond1["radInts_max_cond"], 
        data_cond2["radInts_max_cond"], 
        equal_var=True
        )
    
    # Append
    data = {
        
        "crops"                 : crops,
        "radInts"               : radInts,
        "radInts_max"           : radInts_max,
        "crops_cond1_avg"       : data_cond1["crops_cond_avg"],
        "radInts_cond1_avg"     : data_cond1["radInts_cond_avg"],
        "radInts_cond1_std"     : data_cond1["radInts_cond_std"],
        "radInts_max_cond1"     : data_cond1["radInts_max_cond"],
        "crops_cond2_avg"       : data_cond2["crops_cond_avg"],
        "radInts_cond2_avg"     : data_cond2["radInts_cond_avg"],
        "radInts_cond2_std"     : data_cond2["radInts_cond_std"],
        "radInts_max_cond2"     : data_cond2["radInts_max_cond"],
        "radInts_max_t_stat"    : radInts_max_t_stat,
        "radInts_max_p_value"   : radInts_max_p_value,
        
        }
                
    return data


#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    t0 = time.time()
    
    # Paths 
    paths = list(data_path.glob("*.czi"))
    
    # Open data
    C1s, C2s = open_czi(paths)
    coords_path = Path(data_path, "coords.csv")
    coords = np.loadtxt(coords_path, delimiter=",").astype(int)
       
    # Analyse
    C1_data = analyse(C1s, coords, crop_size, cond1=ctr, cond2=pad)
    C2_data = analyse(C2s, coords, crop_size, cond1=ctr, cond2=pad)
                
    t1 = time.time()
    print(f"runtime = {t1 - t0:.5f}")
    
#%% Plot

fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(2, 3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# -----------------------------------------------------------------------------

# Plot 1
ax1.set_title("C1 norm. fluo. int.")
color_cond1 = "green"
color_cond2 = "lightgreen"
y_cond1 = C1_data["radInts_cond1_avg"]
err_cond1 = C1_data["radInts_cond1_std"]
x_cond1 = np.arange(len(y_cond1))
y_cond2 = C1_data["radInts_cond2_avg"]
err_cond2 = C1_data["radInts_cond2_std"]
x_cond2 = np.arange(len(y_cond2))
ax1.plot(x_cond1, y_cond1, color=color_cond1, label="control")
ax1.plot(x_cond2, y_cond2, color=color_cond2, label="PAD")
ax1.fill_between(
    x_cond1, y_cond1 - err_cond1, y_cond1 + err_cond1,
    alpha=0.05, color=color_cond1,
    )
ax1.fill_between(
    x_cond2, y_cond2 - err_cond2, y_cond2 + err_cond2,
    alpha=0.05, color=color_cond2,
    )
# ax1.set_ylim(-0.1, 1.1)
ax1.set_xlim(0, 50)
ax1.set_ylabel("norm. fluo. int.")
ax1.set_xlabel("Distance (pixels)")
ax1.legend(loc="lower left")

# Plot 2
ax2.set_title("C1 dist. of max. fluo. int.")
ax2.boxplot(C1_data["radInts_max_cond1"], positions=[1], 
    widths=0.6, showfliers=False)
ax2.boxplot(C1_data["radInts_max_cond2"], positions=[2], 
    widths=0.6, showfliers=False)
ax2.set_xticks([1, 2], ["control", "PAD"])
ax2.set_ylabel("Distance (pixels)")

# Plot 3
ax3.set_title("C2 norm. fluo. int.")
color_cond1 = "darkmagenta"
color_cond2 = "magenta"
y_cond1 = C2_data["radInts_cond1_avg"]
err_cond1 = C2_data["radInts_cond1_std"]
x_cond1 = np.arange(len(y_cond1))
y_cond2 = C2_data["radInts_cond2_avg"]
err_cond2 = C2_data["radInts_cond2_std"]
x_cond2 = np.arange(len(y_cond2))
ax3.plot(x_cond1, y_cond1, color=color_cond1, label="control")
ax3.plot(x_cond2, y_cond2, color=color_cond2, label="PAD")
ax3.fill_between(
    x_cond1, y_cond1 - err_cond1, y_cond1 + err_cond1,
    alpha=0.05, color=color_cond1,
    )
ax3.fill_between(
    x_cond2, y_cond2 - err_cond2, y_cond2 + err_cond2,
    alpha=0.05, color=color_cond2,
    )
# ax1.set_ylim(-0.1, 1.1)
ax3.set_xlim(0, 50)
ax3.set_ylabel("Norm. radial fluo. int.")
ax3.set_xlabel("Distance (pixels)")
ax3.legend(loc="lower left")

# Plot 4
ax4.set_title("C2 dist. of max. fluo. int.")
ax4.boxplot(C2_data["radInts_max_cond1"], positions=[1], 
    widths=0.6, showfliers=False)
ax4.boxplot(C2_data["radInts_max_cond2"], positions=[2], 
    widths=0.6, showfliers=False)
ax4.set_xticks([1, 2], ["control", "PAD"])
ax4.set_ylabel("Distance (pixels)")

plt.tight_layout()
plt.show()