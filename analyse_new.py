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
        dmax = np.argmax(radInt)
        prom = np.max(radInt) - 1
        width = np.sum(radInt > (prom / 2) + 1)
        return dmax, prom, width
   
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
    crops, radInts, dmaxs, proms, widths = [], [], [], [], []
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
                dmax, prom, width = get_peak_info(radInt)
                                
                # Append
                crops.append(crop)
                radInts.append(radInt)
                dmaxs.append(dmax)
                proms.append(prom)
                widths.append(width)
                
    # Average patient data
    pAvg_crops, pStd_crops = average_pData(crops, patients)
    pAvg_radInts, pStd_radInts = average_pData(radInts, patients)
    pAvg_dmaxs, pStd_dmaxs = average_pData(dmaxs, patients)
    pAvg_proms, pStd_proms = average_pData(proms, patients)
    pAvg_widths, pStd_widths = average_pData(widths, patients)
    
    # Average condition data (from patient data)
    c0p_crops, c0pAvg_crops, c0pStd_crops = average_cpData(
        pAvg_crops, p_conditions, cond=0)
    c1p_crops, c1pAvg_crops, c1pStd_crops = average_cpData(
        pAvg_crops, p_conditions, cond=1)
    c0p_radInts, c0pAvg_radInts, c0pStd_radInts = average_cpData(
        pAvg_radInts, p_conditions, cond=0)
    c1p_radInts, c1pAvg_radInts, c1pStd_radInts = average_cpData(
        pAvg_radInts, p_conditions, cond=1)
    c0p_dmaxs, c0pAvg_dmaxs, c0pStd_dmaxs = average_cpData(
        pAvg_dmaxs, p_conditions, cond=0)
    c1p_dmaxs, c1pAvg_dmaxs, c1pStd_dmaxs = average_cpData(
        pAvg_dmaxs, p_conditions, cond=1)
    c0p_proms, c0pAvg_proms, c0pStd_proms = average_cpData(
        pAvg_proms, p_conditions, cond=0)
    c1p_proms, c1pAvg_proms, c1pStd_proms = average_cpData(
        pAvg_proms, p_conditions, cond=1)
    c0p_widths, c0pAvg_widths, c0pStd_widths = average_cpData(
        pAvg_widths, p_conditions, cond=0)
    c1p_widths, c1pAvg_widths, c1pStd_widths = average_cpData(
        pAvg_widths, p_conditions, cond=1)
    
    # Append
    data = {
        
        "patients"      : patients,
        "conditions"    : conditions,
        "crops"         : crops,
        "radInts"       : radInts,
        "dmaxs"         : dmaxs,
        "proms"         : proms,
        "widths"        : widths,
        
        "p_patients"   : p_patients,   "p_conditions" : p_conditions,
        "pAvg_crops"   : pAvg_crops,   "pStd_crops"   : pStd_crops,
        "pAvg_radInts" : pAvg_radInts, "pStd_radInts" : pStd_radInts,
        "pAvg_dmaxs"   : pAvg_dmaxs,   "pStd_dmaxs"   : pStd_dmaxs,
        "pAvg_proms"   : pAvg_proms,   "pStd_proms"   : pStd_proms,
        "pAvg_widths"  : pAvg_widths,  "pStd_widths"  : pStd_widths,
        
        "c0p_crops"   : c0p_crops,   "c0pAvg_crops"   : c0pAvg_crops,   "c0pStd_crops"   : c0pStd_crops,
        "c0p_radInts" : c0p_radInts, "c0pAvg_radInts" : c0pAvg_radInts, "c0pStd_radInts" : c0pStd_radInts,
        "c0p_dmaxs"   : c0p_dmaxs,   "c0pAvg_dmaxs"   : c0pAvg_dmaxs,   "c0pStd_dmaxs"   : c0pStd_dmaxs,
        "c0p_proms"   : c0p_proms,   "c0pAvg_proms"   : c0pAvg_proms,   "c0pStd_proms"   : c0pStd_proms,
        "c0p_widths"  : c0p_widths,  "c0pAvg_widths"  : c0pAvg_widths,  "c0pStd_widths"  : c0pStd_widths,
        
        "c1p_crops"   : c1p_crops,   "c1pAvg_crops"   : c1pAvg_crops,   "c1pStd_crops"   : c1pStd_crops,
        "c1p_radInts" : c1p_radInts, "c1pAvg_radInts" : c1pAvg_radInts, "c1pStd_radInts" : c1pStd_radInts,
        "c1p_dmaxs"   : c1p_dmaxs,   "c1pAvg_dmaxs"   : c1pAvg_dmaxs,   "c1pStd_dmaxs"   : c1pStd_dmaxs,
        "c1p_proms"   : c1p_proms,   "c1pAvg_proms"   : c1pAvg_proms,   "c1pStd_proms"   : c1pStd_proms,
        "c1p_widths"  : c1p_widths,  "c1pAvg_widths"  : c1pAvg_widths,  "c1pStd_widths"  : c1pStd_widths,
        
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

#     global radInts, dmax, dmax_cond1, dmax_cond2, tmp_nan

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
    
#     # (.csv) dmax 
#     dmax_cond1 = np.stack(data["dmax_cond1"]).astype(float)
#     dmax_cond2 = np.stack(data["dmax_cond2"]).astype(float)
#     max_len = np.maximum(len(dmax_cond1), len(dmax_cond2))
#     dmax = np.full((max_len, 4), np.nan)
#     dmax[:len(dmax_cond1), 0] = dmax_cond1
#     dmax[:len(dmax_cond2), 1] = dmax_cond2
#     dmax[0, 2] = data["dmax_t_stat"]
#     dmax[0, 3] = data["dmax_p_value"]
#     dmax = pd.DataFrame(dmax, columns=[
#         "cond1", "cond2", "t_stat", "p_value"])
#     dmax.to_csv(
#         Path(data_path, f"{tag}_dmax.csv"), 
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

# Patient results dmax
pResults_dmax = np.array([
    data["p_conditions"], data["p_patients"],
    data["pAvg_dmaxs"], data["pStd_dmaxs"],
    ]).T

# Condition results radInts
cpResults_radInts = np.array([
    data["c0pAvg_radInts"], data["c0pStd_radInts"],
    data["c1pAvg_radInts"], data["c1pStd_radInts"],
    ]).T

#%% Plot ----------------------------------------------------------------------

fig = plt.figure(figsize=(8, 8))

gs = fig.add_gridspec(2, 3)
ax1 = fig.add_subplot(gs[:2, :2])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 1])
ax4 = fig.add_subplot(gs[2, 2])
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
ax3.set_title("Control avg. img.")
ax3.imshow(C1_data["crops_cond1_avg"], cmap="Greens")
ax3.set_ylabel("y")
ax3.set_xlabel("x")

# Plot 3
ax4.set_title("PAD avg. img.")
ax4.imshow(C1_data["crops_cond2_avg"], cmap="Greens")
ax4.set_ylabel("y")
ax4.set_xlabel("x")

# Plot 2
ax2.set_title(
    f"Dist. of max. fluo. int.\n"
    # f"p value = {C1_dmax_p_value:.2e}"
    )
c0p_dmaxs = data["c0p_dmaxs"]
c1p_dmaxs = data["c1p_dmaxs"]
ax2.boxplot(c0p_dmaxs, positions=[1], widths=0.6, showfliers=False)
ax2.boxplot(c1p_dmaxs, positions=[2], widths=0.6, showfliers=False)
ax2.scatter(
    [1] * len(c0p_dmaxs), c0p_dmaxs, color=c0_color, alpha=0.7, label="Control")
ax2.scatter(
    [2] * len(c1p_dmaxs), c1p_dmaxs, color=c1_color, alpha=0.7, label="PAD")
ax2.set_xticks([1, 2], ["control", "PAD"])
ax2.set_ylabel("Distance (pixels)")

# Plot 3
ax3.set_title(
    f"Prom. of max. fluo. int.\n"
    # f"p value = {C1_dmax_p_value:.2e}"
    )
c0p_proms = data["c0p_proms"]
c1p_proms = data["c1p_proms"]
ax3.boxplot(c0p_proms, positions=[1], widths=0.6, showfliers=False)
ax3.boxplot(c1p_proms, positions=[2], widths=0.6, showfliers=False)
ax3.scatter(
    [1] * len(c0p_proms), c0p_proms, color=c0_color, alpha=0.7, label="Control")
ax3.scatter(
    [2] * len(c1p_proms), c1p_proms, color=c1_color, alpha=0.7, label="PAD")
ax3.set_xticks([1, 2], ["control", "PAD"])
ax3.set_ylabel("Prominence (A.U.)")

# Plot 4
ax4.set_title(
    f"Width of max. fluo. int.\n"
    # f"p value = {C1_dmax_p_value:.2e}"
    )
c0p_widths = data["c0p_widths"]
c1p_widths = data["c1p_widths"]
ax4.boxplot(c0p_widths, positions=[1], widths=0.6, showfliers=False)
ax4.boxplot(c1p_widths, positions=[2], widths=0.6, showfliers=False)
ax4.scatter(
    [1] * len(c0p_widths), c0p_widths, color=c0_color, alpha=0.7, label="Control")
ax4.scatter(
    [2] * len(c1p_widths), c1p_widths, color=c1_color, alpha=0.7, label="PAD")
ax4.set_xticks([1, 2], ["control", "PAD"])
ax4.set_ylabel("Width (pixels)")

plt.tight_layout()
plt.show()
