#%% Info ----------------------------------------------------------------------

#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from pathlib import Path
from scipy.interpolate import griddata
from skimage.feature import peak_local_max

from keras.models import load_model

from utils import ImageData, get_patch, get_generator

#%% Parameters ----------------------------------------------------------------

image_number = 0

patchSize = 160
patchStep = 10
patchFreq = 1
rSeed = 0
batchSize = 64

#%% Initialize ----------------------------------------------------------------

data_path = Path('D:/local_Masschelein/data')
model_path = Path('D:/local_Masschelein/data/model')

# Get paths
czi_paths = []
for path in data_path.iterdir():
    if path.suffix == '.czi':
        czi_paths.append(path)

#%% Predict -------------------------------------------------------------------

# Load model
model = load_model(Path(model_path, 'model.h5'))

# Get image_data
my_image = ImageData(czi_paths[image_number])

# Get patch_data
patch_data = get_patch(my_image, patchSize, patchStep, patchFreq)
patch = [data[0] for data in patch_data]

# Get generators
generator = get_generator(
    patch, label=None, batchSize=batchSize
    )

# Get predictions
steps = len(patch) // batchSize + (len(patch) % batchSize != 0)
predictions = model.predict(generator, steps=steps)

# Generate prediction map (pMap)
pMap = np.zeros_like(my_image.image) * np.nan
xCtr = [data[1] for data in patch_data]
yCtr = [data[2] for data in patch_data]
for i, (x, y) in enumerate(zip(xCtr, yCtr)):
    pMap[y, x] = predictions[i]
y, x = np.mgrid[0:pMap.shape[0], 0:pMap.shape[1]]
mask = ~np.isnan(pMap)
pMap = griddata((y[mask], x[mask]), pMap[mask], (y, x), method='cubic')

# #
# coords = peak_local_max(
#     pMap, min_distance=patchSize, threshold_abs=0.2
#     ).astype(int)
    
# Display
viewer = napari.Viewer()
viewer.add_image(my_image.image)
viewer.add_image(pMap)
# points_layer = viewer.add_points(
#     coords, 
#     size=patchSize,
#     edge_width=0.1,
#     edge_color='red',
#     face_color='transparent',
#     opacity = 0.5,
#     )