#%% Imports -------------------------------------------------------------------

import re
import csv
import napari
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from pylibCZIrw import czi as pyczi
from scipy.ndimage import distance_transform_edt

#%% Parameters ----------------------------------------------------------------

cropSize = 160
ctrl = [4, 6, 7, 11]
PAD = [8, 12, 13, 17, 18, 19]

#%% Initialize ----------------------------------------------------------------

data_path = Path('D:/local_Masschelein/data')
csv_path = Path('D:/local_Masschelein/data/model/valid_coords.csv')
       
# Open csv file
with open(str(csv_path), 'r') as f:
    reader = csv.reader(f)
    valid_coords = [
        tuple(int(x) if x.isdigit() else x for x in row) for row in reader
        ]

#%% Process -------------------------------------------------------------------

# Extract crops
crop_data = []
for data in valid_coords:
    
    # Get variables
    name = data[0]
    pad = cropSize // 2
    y = data[1]; x = data[2]
    path = Path(data_path) / data[0]
    
    # Get condition
    match = re.match(r'(\d+)', name)
    if match:
        number = int(match.group(1))
        if number in ctrl:
            cond = 'ctrl'
        elif number in PAD:
            cond = 'PAD'
    
    # Open and crop images
    with pyczi.open_czi(str(path)) as czidoc:
        img = czidoc.read(plane={'T': 0, 'Z': 0, 'C': 0}).squeeze()
    img = np.pad(img, pad_width=pad)
    crop = img[y:y+cropSize, x:x+cropSize]
    
    # Get radial average images
    nY, nX = crop.shape
    edm = np.zeros((nY, nX), dtype=float)
    edm[nY//2, nX//2] = 1
    edm = distance_transform_edt(1 - edm).astype(int)
    unique, counts = np.unique(edm, return_counts=True)
    bin_means = np.bincount(edm.astype(int).ravel(), weights=crop.ravel())
    mVal = bin_means / counts
    mapper = np.zeros(edm.max().astype(int) + 1)
    mapper[unique.astype(int)] = mVal
    rAvg2D = mapper[edm.astype(int)]
    rAvg1D = rAvg2D[:, pad]
    
    # Append crop_data
    crop_data.append((name, cond, crop, rAvg2D, rAvg1D))  
    
#%% Results and displays ------------------------------------------------------

# Extract variables
names = [data[0] for data in crop_data]
conds = [data[1] for data in crop_data]
crops = [data[2] for data in crop_data]
rAvgs2D = [data[3] for data in crop_data] 

# Napari displays
viewer = napari.Viewer()
rAvg_layer = viewer.add_image(np.stack(rAvgs2D), name='rAvg')
crop_layer = viewer.add_image(np.stack(crops), name='crop')
current_slice = viewer.dims.current_step[0]
viewer.text_overlay.text = f'{names[current_slice]} / {conds[current_slice]}'
viewer.text_overlay.visible = True
def update_text_label(event):
    current_slice = viewer.dims.current_step[0]
    viewer.text_overlay.text = f'{names[current_slice]} / {conds[current_slice]}'
viewer.dims.events.current_step.connect(update_text_label)

# Plot average 1D profile per condition
rAvgs1D_ctrl = [data[4] for data in crop_data if data[1] == 'ctrl']
rAvgs1D_PAD = [data[4] for data in crop_data if data[1] == 'PAD']
plt.plot(np.mean(rAvgs1D_ctrl, axis=0), label='Ctrl')
plt.plot(np.mean(rAvgs1D_PAD, axis=0), label='PAD')
plt.legend()
