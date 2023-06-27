#%% Imports -------------------------------------------------------------------

import numpy as np
from pathlib import Path
from pylibCZIrw import czi as pyczi
from skimage.segmentation import clear_border
from skimage.filters import gaussian, threshold_triangle
from skimage.measure import label, regionprops
from skimage.morphology import (
    binary_erosion, binary_dilation, remove_small_objects
    )

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

# Get paths
czi_paths = []
for czi_path in data_path.iterdir():
    if czi_path.is_file():
        czi_paths.append(czi_path)

#%% Process -------------------------------------------------------------------

merged_data = []
for czi_path in czi_paths:

    # Open images
    with pyczi.open_czi(str(czi_path)) as czidoc:
        C1 = czidoc.read(plane={'T': 0, 'Z': 0, 'C': 0}).squeeze()
        C2 = czidoc.read(plane={'T': 0, 'Z': 0, 'C': 1}).squeeze()
    
    # Detect CD31 positive cells
    C2 = gaussian(C2, sigma, preserve_range=True)
    tresh = threshold_triangle(C2)
    mask = C2 > tresh * thresh_coeff
    mask = remove_small_objects(mask, min_size=512)
    mask = clear_border(mask)
    labels = label(mask)
    
    # Get object info
    for i, prop in enumerate(regionprops(labels)):
        if prop['major_axis_length'] / prop['minor_axis_length'] > max_elongation:
            labels[labels==i+1] = 0
    
    # Remove wrong objects
    mask_filt = labels > 0
    
    # Append data list
    merged_data.append((
        C1, C2, labels, mask, mask_filt
        ))
    
#%% Display -------------------------------------------------------------------

C1 = np.stack([data[0] for data in merged_data])
C2 = np.stack([data[1] for data in merged_data])
C3 = C1 * C2
mask = np.stack(
    [binary_dilation(data[3]) ^ binary_erosion(data[3]) for data in merged_data])
mask_filt = np.stack(
    [binary_dilation(data[4]) ^ binary_erosion(data[4]) for data in merged_data])

import napari
viewer = napari.Viewer()
viewer.add_image(C1)
viewer.add_image(C2)
viewer.add_image(C3)
viewer.add_image(mask, blending='additive', colormap='red')
viewer.add_image(mask_filt, blending='additive', colormap='gray')