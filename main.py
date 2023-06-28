#%% Imports -------------------------------------------------------------------

import csv
import cv2
import numpy as np
from skimage import io 
from pathlib import Path
from pylibCZIrw import czi as pyczi
from skimage.feature import peak_local_max
from skimage.filters import gaussian

#%% Comments ------------------------------------------------------------------

''' C1 = laminin (cell membrane marker) '''
''' C2 = CD31 (endothelial cell marker) '''

#%% Parameters ----------------------------------------------------------------

test = 1

sigma = 3
thresh_coeff = 1.25
min_size = 512
max_elongation = 2 

#%% Initialize ----------------------------------------------------------------

data_path = Path('D:\local_Masschelein\data')
model_path = Path('D:\local_Masschelein\data\model')

# Get paths
czi_paths = []
for czi_path in data_path.iterdir():
    if czi_path.is_file():
        czi_paths.append(czi_path)
    
#%% Extract ROIs --------------------------------------------------------------

# Parameters
sigma = 3
minDist = 50
minProm = 0.15
roiSize = 160

mData = []
for czi_path in czi_paths:
    
    # Open images
    with pyczi.open_czi(str(czi_path)) as czidoc:
        C1 = czidoc.read(plane={'T': 0, 'Z': 0, 'C': 0}).squeeze()
        C2 = czidoc.read(plane={'T': 0, 'Z': 0, 'C': 1}).squeeze()
                
    # Find local maxima
    process = gaussian(C1.astype(float) * C2.astype(float), sigma).astype('float32')
    process = (process - np.min(process)) / (np.max(process) - np.min(process))
    coords = peak_local_max(
        process, min_distance=minDist, threshold_abs=minProm
        ).astype(int)
    
    # Crop ROIs and make display
    C1_ROIs, C2_ROIs = [], []
    display = np.zeros_like(C1, dtype='uint16')
    for coord in coords:
        hSize = roiSize // 2
        y, x = coord[0], coord[1] 
        if (y - hSize > 0 and y + hSize < C1.shape[0] and
            x - hSize > 0 and x + hSize < C1.shape[1]):              
            C1_ROIs.append(
                C1[y - hSize: y + hSize, x - hSize: x + hSize],
                )
            C2_ROIs.append(
                C2[y - hSize: y + hSize, x - hSize: x + hSize],
                )
            cv2.rectangle(display,
                (x - hSize, y - hSize),
                (x + hSize, y + hSize),
                (65635), thickness=2,
                )
        
    # Append mData
    mData.append((
        C1, C2, process, coords, C1_ROIs, C2_ROIs, display
        ))
    
#%% Save data -----------------------------------------------------------------
    
# Extract
display = np.stack((
    np.stack([data[0] for data in mData]),
    np.stack([data[1] for data in mData]),
    np.stack([data[6] for data in mData]),
    ), axis=1)
C1_ROIs = np.stack([C1_ROI for data in mData for C1_ROI in data[4]])
C2_ROIs = np.stack([C2_ROI for data in mData for C2_ROI in data[5]])

# Save
io.imsave(
    Path(model_path / 'display.tif'), display, 
    check_contrast=False, imagej=True,
    metadata={'axes': 'TCYX'}
    )
io.imsave(
    Path(model_path / 'C1_ROIs.tif'), C1_ROIs, 
    check_contrast=False, imagej=True,
    metadata={'axes': 'TYX'}
    )
io.imsave(
    Path(model_path / 'C2_ROIs.tif'), C2_ROIs, 
    check_contrast=False, imagej=True,
    metadata={'axes': 'TYX'}
    )

# # Make a new class.csv
# csv_path = model_path / 'class.csv'
# with open(csv_path, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     for i in range(1, 1144):
#         writer.writerow([i, 0])
