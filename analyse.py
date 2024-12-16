#%% Imports -------------------------------------------------------------------

import re
import time
import numpy as np
import pandas as pd
from skimage import io
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
        data_cond1["radInts_max_cond"], data_cond2["radInts_max_cond"], 
        equal_var=True
        )
    
    # Append
    data = {
        
        "crops"               : crops,
        "radInts"             : radInts,
        "radInts_max"         : radInts_max,
        
        "crops_cond1_avg"     : data_cond1["crops_cond_avg"],
        "radInts_cond1_avg"   : data_cond1["radInts_cond_avg"],
        "radInts_cond1_std"   : data_cond1["radInts_cond_std"],
        "radInts_max_cond1"   : data_cond1["radInts_max_cond"],
        
        "crops_cond2_avg"     : data_cond2["crops_cond_avg"],
        "radInts_cond2_avg"   : data_cond2["radInts_cond_avg"],
        "radInts_cond2_std"   : data_cond2["radInts_cond_std"],
        "radInts_max_cond2"   : data_cond2["radInts_max_cond"],
        
        "radInts_max_t_stat"  : radInts_max_t_stat,
        "radInts_max_p_value" : radInts_max_p_value,
        
        }
    
    return data

def save(data, data_path, tag="C1"):

    global radInts, radInts_max, radInts_max_cond1, radInts_max_cond2, tmp_nan

    # (.tif) Image 
    io.imsave(
        Path(data_path, f"{tag}_crops_cond1_avg.tif"),
        data["crops_cond1_avg"].astype("float32"), check_contrast=False,
        )
    io.imsave(
        Path(data_path, f"{tag}_crops_cond2_avg.tif"),
        data["crops_cond2_avg"].astype("float32"), check_contrast=False,
        )
    
    # (.csv) radInts 
    radInts = np.vstack((
        data["radInts_cond1_avg"], data["radInts_cond1_std"],
        data["radInts_cond2_avg"], data["radInts_cond2_std"],
        )).T
    radInts = pd.DataFrame(radInts, columns=[
        "avg. cond1", "std. cond1", "avg. cond2", "std. cond2"])
    radInts.to_csv(
        Path(data_path, f"{tag}_radInts.csv"), 
        index=False, float_format="%.3f"
        )
    
    # (.csv) radInts_max 
    radInts_max_cond1 = np.stack(data["radInts_max_cond1"]).astype(float)
    radInts_max_cond2 = np.stack(data["radInts_max_cond2"]).astype(float)
    max_len = np.maximum(len(radInts_max_cond1), len(radInts_max_cond2))
    radInts_max = np.full((max_len, 4), np.nan)
    radInts_max[:len(radInts_max_cond1), 0] = radInts_max_cond1
    radInts_max[:len(radInts_max_cond2), 1] = radInts_max_cond2
    radInts_max[0, 2] = data["radInts_max_t_stat"]
    radInts_max[0, 3] = data["radInts_max_p_value"]
    radInts_max = pd.DataFrame(radInts_max, columns=[
        "cond1", "cond2", "t_stat", "p_value"])
    radInts_max.to_csv(
        Path(data_path, f"{tag}_radInts_max.csv"), 
        index=False, float_format="%.5f"
        )

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
    
    # Save
    save(C1_data, data_path, tag="C1")
    save(C2_data, data_path, tag="C2")
    
    t1 = time.time()
    print(f"runtime = {t1 - t0:.5f}")
    
#%% Plot ----------------------------------------------------------------------

    fig = plt.figure(figsize=(8, 8))
    
    gs = fig.add_gridspec(4, 3)
    ax1 = fig.add_subplot(gs[:2, 0])
    ax2 = fig.add_subplot(gs[:2, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 2])
    ax5 = fig.add_subplot(gs[2:, 0])
    ax6 = fig.add_subplot(gs[2:, 1])
    ax7 = fig.add_subplot(gs[2, 2])
    ax8 = fig.add_subplot(gs[3, 2])
    
    # -------------------------------------------------------------------------
    
    # Plot options
    C1_color_cond1 = "green"
    C1_color_cond2 = "lightgreen"
    C2_color_cond1 = "darkmagenta"
    C2_color_cond2 = "magenta"
    
    # Extract data
    C1_radInts_max_t_stat = C1_data["radInts_max_t_stat"]
    C1_radInts_max_p_value = C1_data["radInts_max_p_value"]
    C2_radInts_max_t_stat = C2_data["radInts_max_t_stat"]
    C2_radInts_max_p_value = C2_data["radInts_max_p_value"]
    
    # -------------------------------------------------------------------------
    
    # Plot 1
    ax1.set_title("C1 norm. fluo. int.")
    y_cond1 = C1_data["radInts_cond1_avg"]
    err_cond1 = C1_data["radInts_cond1_std"]
    x_cond1 = np.arange(len(y_cond1))
    y_cond2 = C1_data["radInts_cond2_avg"]
    err_cond2 = C1_data["radInts_cond2_std"]
    x_cond2 = np.arange(len(y_cond2))
    ax1.plot(x_cond1, y_cond1, color=C1_color_cond1, label="control")
    ax1.plot(x_cond2, y_cond2, color=C1_color_cond2, label="PAD")
    ax1.fill_between(
        x_cond1, y_cond1 - err_cond1, y_cond1 + err_cond1,
        alpha=0.05, color=C1_color_cond1)
    ax1.fill_between(
        x_cond2, y_cond2 - err_cond2, y_cond2 + err_cond2,
        alpha=0.05, color=C1_color_cond2)
    ax1.set_xlim(0, 50)
    ax1.set_ylabel("norm. fluo. int.")
    ax1.set_xlabel("Distance (pixels)")
    ax1.legend(loc="lower left")
    
    # Plot 2
    ax2.set_title(
        f"C1 dist. of max. fluo. int.\n"
        f"p value = {C1_radInts_max_p_value:.2e}"
        )
    ax2.boxplot(C1_data["radInts_max_cond1"], positions=[1], 
        widths=0.6, showfliers=False)
    ax2.boxplot(C1_data["radInts_max_cond2"], positions=[2], 
        widths=0.6, showfliers=False)
    ax2.set_xticks([1, 2], ["control", "PAD"])
    ax2.set_ylabel("Distance (pixels)")
    
    # Plot 3
    ax3.set_title("Control avg. img.")
    ax3.imshow(C1_data["crops_cond1_avg"], cmap="Greens")
    ax3.set_ylabel("y")
    ax3.set_xlabel("x")
    
    # Plot 4
    ax4.set_title("PAD avg. img.")
    ax4.imshow(C1_data["crops_cond2_avg"], cmap="Greens")
    ax4.set_ylabel("y")
    ax4.set_xlabel("x")
    
    # -------------------------------------------------------------------------
    
    # Plot 5
    ax5.set_title("C2 norm. fluo. int.")
    y_cond1 = C2_data["radInts_cond1_avg"]
    err_cond1 = C2_data["radInts_cond1_std"]
    x_cond1 = np.arange(len(y_cond1))
    y_cond2 = C2_data["radInts_cond2_avg"]
    err_cond2 = C2_data["radInts_cond2_std"]
    x_cond2 = np.arange(len(y_cond2))
    ax5.plot(x_cond1, y_cond1, color=C2_color_cond1, label="control")
    ax5.plot(x_cond2, y_cond2, color=C2_color_cond2, label="PAD")
    ax5.fill_between(
        x_cond1, y_cond1 - err_cond1, y_cond1 + err_cond1,
        alpha=0.05, color=C2_color_cond1)
    ax5.fill_between(
        x_cond2, y_cond2 - err_cond2, y_cond2 + err_cond2,
        alpha=0.05, color=C2_color_cond2)
    ax5.set_xlim(0, 50)
    ax5.set_ylabel("Norm. radial fluo. int.")
    ax5.set_xlabel("Distance (pixels)")
    ax5.legend(loc="lower left")
    
    # Plot 6
    ax6.set_title(
        f"C2 dist. of max. fluo. int.\n"
        f"p value = {C2_radInts_max_p_value:.2e}"
        )
    ax6.boxplot(C2_data["radInts_max_cond1"], positions=[1], 
        widths=0.6, showfliers=False)
    ax6.boxplot(C2_data["radInts_max_cond2"], positions=[2], 
        widths=0.6, showfliers=False)
    ax6.set_xticks([1, 2], ["control", "PAD"])
    ax6.set_ylabel("Distance (pixels)")
    
    # Plot 7
    ax7.set_title("Control avg. img.")
    ax7.imshow(C2_data["crops_cond1_avg"], cmap="Purples")
    ax7.set_ylabel("y")
    ax7.set_xlabel("x")
    
    # Plot 8
    ax8.set_title("PAD avg. img.")
    ax8.imshow(C2_data["crops_cond2_avg"], cmap="Purples")
    ax8.set_ylabel("y")
    ax8.set_xlabel("x")
    
    plt.tight_layout()
    plt.savefig(Path(data_path, "plot.png"), dpi=300, bbox_inches='tight')
    plt.show()