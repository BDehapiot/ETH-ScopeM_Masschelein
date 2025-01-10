#%% Imports -------------------------------------------------------------------

import re
import time
import shutil
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
from scipy.signal import find_peaks, peak_prominences, peak_widths

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path('D:/local_Masschelein/data')

# Parameters
crop_size = 120
ctr = [1, 3, 4, 5, 6, 7, 9, 11]
pad = [2, 8, 10, 12, 13, 14, 17, 18, 19]

#%% Function(s) ---------------------------------------------------------------

def analyse(paths, imgs, coords, crop_size, cond0=ctr, cond1=pad):
    
    # Nested function(s) ------------------------------------------------------
    
    def get_peak_info(radInt):
        radInt_max = np.argmax(radInt)
        radInt_prom = np.max(radInt) - 1
        radInt_width = np.sum(radInt > (radInt_prom / 2) + 1)
        return radInt_max, radInt_prom, radInt_width
   
    def average_pData(data, patients):
        pAvg_data, pStd_data = [], []
        for patient in np.unique(patients):
            p_data = [dat for i, dat in enumerate(data) 
                if patients[i] == patient]
            pAvg_data.append(np.nanmean(np.stack(p_data), axis=0))
            pStd_data.append(np.nanstd(np.stack(p_data), axis=0))
        return pAvg_data, pStd_data
    
    def average_cpData(data, p_conditions, cond=0):
        cp_data = [dat for i, dat in enumerate(data) if p_conditions[i] == cond]
        cpAvg_data = np.nanmean(np.stack(cp_data), axis=0)
        cpStd_data = np.nanstd(np.stack(cp_data), axis=0)
        return cp_data, cpAvg_data, cpStd_data
    
    # Execute -----------------------------------------------------------------

    # Get info
    patients, conditions, p_patients, p_conditions = [], [], [], []
    for i in coords[:, 0]:
        match = re.match(r"(\d+)\.\d+\.\w+", paths[i].name)
        patient = int(match.group(1))
        if patient in cond0: conditions.append(0)
        if patient in cond1: conditions.append(1)
        patients.append(patient)
    for patient in np.unique(patients):
        p_patients.append(patient)
        if patient in cond0: p_conditions.append(0)
        if patient in cond1: p_conditions.append(1)
        
    # Get edt
    edt = np.zeros((crop_size, crop_size))
    edt[crop_size // 2, crop_size // 2] = 1
    edt = distance_transform_edt(1 - edt).astype(int)
    unique, counts = np.unique(edt, return_counts=True) 
    
    # Crop & analyse data 
    crops, radInts, radInts_max, radInts_prom, radInts_width = [], [], [], [], []
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
                radInt_max, radInt_prom, radInt_width = get_peak_info(radInt)
                                
                # Append
                crops.append(crop)
                radInts.append(radInt)
                radInts_max.append(radInt_max)
                radInts_prom.append(radInt_prom)
                radInts_width.append(radInt_width)
                
    # Average patient data
    pAvg_crops, pStd_crops = average_pData(crops, patients)
    pAvg_radInts, pStd_radInts = average_pData(radInts, patients)
    pAvg_radInts_max, pStd_radInts_max = average_pData(radInts_max, patients)
    
    # Average condition data (from patient data)
    c0p_crops, c0pAvg_crops, c0pStd_crops = average_cpData(
        pAvg_crops, p_conditions, cond=0)
    c1p_crops, c1pAvg_crops, c1pStd_crops = average_cpData(
        pAvg_crops, p_conditions, cond=1)
    c0p_radInts, c0pAvg_radInts, c0pStd_radInts = average_cpData(
        pAvg_radInts, p_conditions, cond=0)
    c1p_radInts, c1pAvg_radInts, c1pStd_radInts = average_cpData(
        pAvg_radInts, p_conditions, cond=1)
    c0p_radInts_max, c0pAvg_radInts_max, c0pStd_radInts_max = average_cpData(
        pAvg_radInts_max, p_conditions, cond=0)
    c1p_radInts_max, c1pAvg_radInts_max, c1pStd_radInts_max = average_cpData(
        pAvg_radInts_max, p_conditions, cond=1)
    
    # Append
    data = {
        
        "patients"      : patients,
        "conditions"    : conditions,
        "crops"         : crops,
        "radInts"       : radInts,
        "radInts_max"   : radInts_max,
        "radInts_prom"  : radInts_prom,
        "radInts_width" : radInts_width,
        
        "p_patients"       : p_patients,
        "p_conditions"     : p_conditions,
        "pAvg_crops"       : pAvg_crops,
        "pStd_crops"       : pStd_crops,
        "pAvg_radInts"     : pAvg_radInts,
        "pStd_radInts"     : pStd_radInts,
        "pAvg_radInts_max" : pAvg_radInts_max,
        "pStd_radInts_max" : pStd_radInts_max,
        
        "c0p_crops"          : c0p_crops,
        "c0pAvg_crops"       : c0pAvg_crops,
        "c0pStd_crops"       : c0pStd_crops,
        "c0p_radInts"        : c0p_radInts,
        "c0pAvg_radInts"     : c0pAvg_radInts,
        "c0pStd_radInts"     : c0pStd_radInts,
        "c0p_radInts_max"    : c0p_radInts_max,
        "c0pAvg_radInts_max" : c0pAvg_radInts_max,
        "c0pStd_radInts_max" : c0pStd_radInts_max,
        
        "c1p_crops"          : c1p_crops,
        "c1pAvg_crops"       : c1pAvg_crops,
        "c1pStd_crops"       : c1pStd_crops,
        "c1p_radInts"        : c1p_radInts,
        "c1pAvg_radInts"     : c1pAvg_radInts,
        "c1pStd_radInts"     : c1pStd_radInts,
        "c1p_radInts_max"    : c1p_radInts_max,
        "c1pAvg_radInts_max" : c1pAvg_radInts_max,
        "c1pStd_radInts_max" : c1pStd_radInts_max,
        
        }
    
    return data

def save(data, data_path):
    
    save_path = data_path / "results"
    
    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(exist_ok=True)
    
    pAvg_radInts = np.vstack((data["pAvg_crops"])).T
                              
    pass

# def save(data, data_path, tag="C1"):

#     global radInts, radInts_max, radInts_max_cond1, radInts_max_cond2, tmp_nan

#     # (.tif) Image 
#     io.imsave(
#         Path(data_path, f"{tag}_crops_cond1_avg.tif"),
#         data["crops_cond1_avg"].astype("float32"), check_contrast=False,
#         )
#     io.imsave(
#         Path(data_path, f"{tag}_crops_cond2_avg.tif"),
#         data["crops_cond2_avg"].astype("float32"), check_contrast=False,
#         )
    
#     # (.csv) radInts 
#     radInts = np.vstack((
#         data["radInts_cond1_avg"], data["radInts_cond1_std"],
#         data["radInts_cond2_avg"], data["radInts_cond2_std"],
#         )).T
#     radInts = pd.DataFrame(radInts, columns=[
#         "avg. cond1", "std. cond1", "avg. cond2", "std. cond2"])
#     radInts.to_csv(
#         Path(data_path, f"{tag}_radInts.csv"), 
#         index=False, float_format="%.3f"
#         )
    
#     # (.csv) radInts_max 
#     radInts_max_cond1 = np.stack(data["radInts_max_cond1"]).astype(float)
#     radInts_max_cond2 = np.stack(data["radInts_max_cond2"]).astype(float)
#     max_len = np.maximum(len(radInts_max_cond1), len(radInts_max_cond2))
#     radInts_max = np.full((max_len, 4), np.nan)
#     radInts_max[:len(radInts_max_cond1), 0] = radInts_max_cond1
#     radInts_max[:len(radInts_max_cond2), 1] = radInts_max_cond2
#     radInts_max[0, 2] = data["radInts_max_t_stat"]
#     radInts_max[0, 3] = data["radInts_max_p_value"]
#     radInts_max = pd.DataFrame(radInts_max, columns=[
#         "cond1", "cond2", "t_stat", "p_value"])
#     radInts_max.to_csv(
#         Path(data_path, f"{tag}_radInts_max.csv"), 
#         index=False, float_format="%.5f"
#         )
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    t0 = time.time()
    
    # Paths 
    paths = list(data_path.glob("*.czi"))
    
    # Open data
    imgs, _ = open_czi(paths)
    coords_path = Path(data_path, "coords.csv")
    coords = np.loadtxt(coords_path, delimiter=",").astype(int)
       
    # Analyse
    data = analyse(paths, imgs, coords, crop_size, cond0=ctr, cond1=pad)
    
    t1 = time.time()
    print(f"runtime = {t1 - t0:.5f}")

#%% 

# Patient results radInts
avg = np.stack(data["pAvg_radInts"]).T
std = np.stack(data["pStd_radInts"]).T
pResults_radInts = np.empty((avg.shape[0], avg.shape[1] * 2), dtype=avg.dtype)
pResults_radInts[:,  ::2] = avg
pResults_radInts[:, 1::2] = std

# Patient results radInts_max
pResults_radInts_max = np.array([
    data["p_conditions"], data["p_patients"],
    data["pAvg_radInts_max"], data["pStd_radInts_max"],
    ]).T

# Condition results radInts
cpResults_radInts = np.array([
    data["c0pAvg_radInts"], data["c0pStd_radInts"],
    data["c1pAvg_radInts"], data["c1pStd_radInts"],
    ]).T

#%% Plot ----------------------------------------------------------------------

fig = plt.figure(figsize=(8, 8))

gs = fig.add_gridspec(4, 2)
ax1 = fig.add_subplot(gs[:2, 0])
ax2 = fig.add_subplot(gs[:2, 1])
# ax3 = fig.add_subplot(gs[0, 2])
# ax4 = fig.add_subplot(gs[1, 2])
# ax5 = fig.add_subplot(gs[2:, 0])
# ax6 = fig.add_subplot(gs[2:, 1])
# ax7 = fig.add_subplot(gs[2, 2])
# ax8 = fig.add_subplot(gs[3, 2])

# -----------------------------------------------------------------------------

# Plot options
c0_color = "green"
c1_color = "magenta"

# -----------------------------------------------------------------------------

# Plot 1
ax1.set_title("norm. radial fluo. int.")

c0pAvg = data["c0pAvg_radInts"]
c1pAvg = data["c1pAvg_radInts"]
c0pStd = data["c0pStd_radInts"]
c1pStd = data["c1pStd_radInts"]
c0x = np.arange(len(c0pAvg)) 
c1x = np.arange(len(c1pAvg)) 

ax1.plot(c0pAvg, color=c0_color, linewidth=2, label="control")
ax1.plot(c1pAvg, color=c1_color, linewidth=2, label="PAD")
ax1.fill_between(
    c0x, c0pAvg - c0pStd, c0pAvg + c0pStd, color=c0_color, alpha=0.05)
ax1.fill_between(
    c1x, c1pAvg - c1pStd, c1pAvg + c1pStd, color=c1_color, alpha=0.05)

ax1.set_xlim(0, 50)
ax1.set_ylabel("norm. radial fluo. int.")
ax1.set_xlabel("Distance (pixels)")
ax1.legend(loc="lower left")

# Plot 2
ax2.set_title(
    f"dist. of max. fluo. int.\n"
    # f"p value = {C1_radInts_max_p_value:.2e}"
    )

c0p_max = data["c0p_radInts_max"]
c1p_max = data["c1p_radInts_max"]

ax2.boxplot(c0p_max, positions=[1], widths=0.6, showfliers=False)
ax2.boxplot(c1p_max, positions=[2], widths=0.6, showfliers=False)
ax2.scatter(
    [1] * len(c0p_max), c0p_max, color=c0_color, alpha=0.7, label="Control")
ax2.scatter(
    [2] * len(c1p_max), c1p_max, color=c1_color, alpha=0.7, label="PAD")

ax2.set_xticks([1, 2], ["control", "PAD"])
ax2.set_ylabel("Distance (pixels)")

plt.tight_layout()
plt.show()
