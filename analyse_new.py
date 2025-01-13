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

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path('D:/local_Masschelein/data')

# Parameters
crop_size = 120
ctr = [1, 3, 4, 5, 6, 7, 9, 11]
pad = [2, 8, 10, 12, 13, 14, 17, 18, 19]

#%% Function : analyse() ------------------------------------------------------

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
    
    # Statistics
    cp_dmaxs_stat, cp_dmaxs_pval = stats.ttest_ind(
        c0p_dmaxs, c1p_dmaxs, equal_var=True)
    cp_proms_stat, cp_proms_pval = stats.ttest_ind(
        c0p_proms, c1p_proms, equal_var=True)
    cp_widths_stat, cp_widths_pval = stats.ttest_ind(
        c0p_widths, c1p_widths, equal_var=True)
    
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
        
        "cp_dmaxs_stat" : cp_dmaxs_stat, "cp_dmaxs_pval" : cp_dmaxs_pval,
        "cp_proms_stat" : cp_proms_stat, "cp_proms_pval" : cp_proms_pval,
        "cp_widths_stat" : cp_widths_stat, "cp_widths_pval" : cp_widths_pval,
        
        }
    
    return data

#%% Function : save() ---------------------------------------------------------

def save(data, data_path):

    # Paths & create saving folder
    save_path = data_path / "results"
    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(exist_ok=True)
    
    # -------------------------------------------------------------------------
    
    # (.tif) Image 
    io.imsave(
        Path(save_path, "c0pAvg_crops.tif"),
        data["c0pAvg_crops"].astype("float32"), check_contrast=False,
        )
    io.imsave(
        Path(save_path, "c1pAvg_crops.tif"),
        data["c1pAvg_crops"].astype("float32"), check_contrast=False,
        )
    
    # -------------------------------------------------------------------------
    
    # Patient results radInts
    avg = np.stack(data["pAvg_radInts"]).T
    std = np.stack(data["pStd_radInts"]).T
    pResults_radInts = np.empty((avg.shape[0], avg.shape[1] * 2), dtype=avg.dtype)
    pResults_radInts[:,  ::2] = avg
    pResults_radInts[:, 1::2] = std

    columns = []
    for i in range(avg.shape[1]):
        columns.append(f"p{data['p_patients'][i]:02d} avg.")
        columns.append(f"p{data['p_patients'][i]:02d} std.")
        
    pResults_radInts = pd.DataFrame(pResults_radInts, columns=columns)
    
    pResults_radInts.to_csv(
        Path(save_path, "pResults_radInts.csv"), 
        index=False, float_format="%.5f"
        )

    # -------------------------------------------------------------------------

    # Patient results peaks
    pResults_peaks = np.array([
        data["pAvg_dmaxs"], data["pStd_dmaxs"],
        data["pAvg_proms"], data["pStd_proms"],
        data["pAvg_widths"], data["pStd_widths"],
        ]).T

    columns = [
        "dmaxs avg.", "dmaxs std",
        "proms avg.", "proms std",
        "widths avg.", "widths std",
        ]

    pResults_peaks = pd.DataFrame(pResults_peaks, columns=columns)
    pResults_peaks.index = data["p_patients"]
    pResults_peaks.index.name = "patient"

    pResults_peaks.to_csv(
        Path(save_path, "pResults_peaks.csv"), 
        index=True, float_format="%.5f"
        )

    # -------------------------------------------------------------------------

    # Condition results radInts
    cpResults_radInts = np.array([
        data["c0pAvg_radInts"], data["c0pStd_radInts"],
        data["c1pAvg_radInts"], data["c1pStd_radInts"],
        ]).T

    columns = ["c0 avg.", "c0 std.", "c1 avg.", "c1 std."]

    cpResults_radInts = pd.DataFrame(cpResults_radInts, columns=columns)
    cpResults_radInts.index = np.arange(0, len(data["c0pAvg_radInts"]))
    cpResults_radInts.index.name = "distance"
    
    cpResults_radInts.to_csv(
        Path(save_path, "cpResults_radInts.csv"), 
        index=True, float_format="%.5f"
        )
        
    # -------------------------------------------------------------------------
    
    # Condition results peaks
    cpResults_peaks = np.array([
        np.hstack((
            data["c0pAvg_dmaxs"], data["c0pStd_dmaxs"],
            data["c1pAvg_dmaxs"], data["c1pStd_dmaxs"],
            data["cp_dmaxs_stat"], data["cp_dmaxs_pval"],
            )),
        np.hstack((
            data["c0pAvg_proms"], data["c0pStd_proms"],
            data["c1pAvg_proms"], data["c1pStd_proms"],
            data["cp_proms_stat"], data["cp_proms_pval"],
            )),    
        np.hstack((
            data["c0pAvg_widths"], data["c0pStd_widths"],
            data["c1pAvg_widths"], data["c1pStd_widths"],
            data["cp_widths_stat"], data["cp_widths_pval"],
            )),
        ])

    columns = ["c0 avg.", "c0 std.", "c1 avg.", "c1 std.", "diff.", "pval."]

    cpResults_peaks = pd.DataFrame(cpResults_peaks, columns=columns)
    cpResults_peaks.index = ["dmaxs", "proms", "widths"]
    cpResults_peaks.index.name = "measure"
    
    cpResults_peaks.to_csv(
        Path(save_path, "cpResults_peaks.csv"), 
        index=True, float_format="%.5f"
        )
    
#%% Function : plot() ---------------------------------------------------------
    
def plot(data, data_path):
    
    fig = plt.figure(figsize=(8, 8))
    
    gs = fig.add_gridspec(3, 3)
    ax1 = fig.add_subplot(gs[:2, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 2])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    ax6 = fig.add_subplot(gs[2, 2])
    
    # -------------------------------------------------------------------------
    
    # Plot options
    c0_color = "green"
    c1_color = "magenta"
    
    # -------------------------------------------------------------------------
    
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
    ax2.set_title("Control avg. img.")
    ax2.imshow(data["c0pAvg_crops"], cmap="Greens")
    ax2.set_ylabel("y")
    ax2.set_xlabel("x")
    
    # Plot 3
    ax3.set_title("PAD avg. img.")
    ax3.imshow(data["c1pAvg_crops"], cmap="Purples")
    ax3.set_ylabel("y")
    ax3.set_xlabel("x")
    
    # Plot 4
    ax4.set_title(
        f"Dist. of max. fluo. int.\n"
        f"p value = {data['cp_dmaxs_pval']:.2e}"
        )
    c0p_dmaxs = data["c0p_dmaxs"]
    c1p_dmaxs = data["c1p_dmaxs"]
    ax4.boxplot(c0p_dmaxs, positions=[1], widths=0.6, showfliers=False)
    ax4.boxplot(c1p_dmaxs, positions=[2], widths=0.6, showfliers=False)
    ax4.scatter(
        [1] * len(c0p_dmaxs), c0p_dmaxs, color=c0_color, alpha=0.7, label="Control")
    ax4.scatter(
        [2] * len(c1p_dmaxs), c1p_dmaxs, color=c1_color, alpha=0.7, label="PAD")
    ax4.set_xticks([1, 2], ["control", "PAD"])
    ax4.set_ylabel("Distance (pixels)")
    
    # Plot 5
    ax5.set_title(
        f"Prom. of max. fluo. int.\n"
        f"p value = {data['cp_proms_pval']:.2e}"
        )
    c0p_proms = data["c0p_proms"]
    c1p_proms = data["c1p_proms"]
    ax5.boxplot(c0p_proms, positions=[1], widths=0.6, showfliers=False)
    ax5.boxplot(c1p_proms, positions=[2], widths=0.6, showfliers=False)
    ax5.scatter(
        [1] * len(c0p_proms), c0p_proms, color=c0_color, alpha=0.7, label="Control")
    ax5.scatter(
        [2] * len(c1p_proms), c1p_proms, color=c1_color, alpha=0.7, label="PAD")
    ax5.set_xticks([1, 2], ["control", "PAD"])
    ax5.set_ylabel("Prominence (A.U.)")
    
    # Plot 6
    ax6.set_title(
        f"Half prom. width\n"
        f"p value = {data['cp_widths_pval']:.2e}"
        )
    c0p_widths = data["c0p_widths"]
    c1p_widths = data["c1p_widths"]
    ax6.boxplot(c0p_widths, positions=[1], widths=0.6, showfliers=False)
    ax6.boxplot(c1p_widths, positions=[2], widths=0.6, showfliers=False)
    ax6.scatter(
        [1] * len(c0p_widths), c0p_widths, color=c0_color, alpha=0.7, label="Control")
    ax6.scatter(
        [2] * len(c1p_widths), c1p_widths, color=c1_color, alpha=0.7, label="PAD")
    ax6.set_xticks([1, 2], ["control", "PAD"])
    ax6.set_ylabel("Width (pixels)")
    
    plt.tight_layout()
    plt.savefig(Path(data_path, "results", "plot.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
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
    
    # Save
    save(data, data_path)
    
    # Plot
    plot(data, data_path)
    
    t1 = time.time()
    print(f"runtime = {t1 - t0:.5f}")