#%% Imports -------------------------------------------------------------------

import csv
import time
import napari
import numpy as np
from pathlib import Path
from magicgui import magicgui
import matplotlib.pyplot as plt
from pylibCZIrw import czi as pyczi
from joblib import Parallel, delayed
from sklearn.preprocessing import minmax_scale
from scipy.ndimage import distance_transform_edt
from sklearn.model_selection import train_test_split

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

import tensorflow as tf
print("Available GPUs:", len(tf.config.list_physical_devices('GPU')))

#%% Comments ------------------------------------------------------------------

''' C1 = laminin (cell membrane marker) '''
''' C2 = CD31 (endothelial cell marker) '''

''' 
To do:
- Clean prediction section
- Solve prediction borders (cannot see objects next to the borders)
- Eventually refactorize the process section
- Since patchSize is set by training find a way to enforce it for prediction
- What about channel 2?
'''

#%% Parameters ----------------------------------------------------------------

selectROIs = False
trainModel = False
patchSize = 160
patchStep = 10
patchFreq = 0.1
maxDist = 20
batchSize = 128

#%% Initialize ----------------------------------------------------------------

data_path = Path('D:/local_Masschelein/data')
model_path = Path('D:/local_Masschelein/data/model')
csv_path = Path('D:/local_Masschelein/data/model/roiCoords.csv')

# Get paths
czi_paths = []
for path in data_path.iterdir():
    if path.suffix == '.czi':
        czi_paths.append(path)

# Open csv file
if not selectROIs:
    with open(str(csv_path), 'r') as f:
        reader = csv.reader(f)
        roiCoords = [
            tuple(int(x) if x.isdigit() else x for x in row) for row in reader
            ]

# Open images
C1s, C2s = [], []
for czi_path in czi_paths:      
    with pyczi.open_czi(str(czi_path)) as czidoc:
        C1s.append(czidoc.read(plane={'T': 0, 'Z': 0, 'C': 0}).squeeze())
        C2s.append(czidoc.read(plane={'T': 0, 'Z': 0, 'C': 1}).squeeze())

#%% Select ROIs ---------------------------------------------------------------

if selectROIs:
    
    # Initialize
    points = {}
    viewer = napari.Viewer()
    hstack = np.stack((np.stack(C1s), np.stack(C2s)), axis=0)

    # Setup image layer
    viewer.add_image(
        hstack, 
        channel_axis=0,
        name=['Laminin', 'CD31'],
        colormap=['green', 'magenta']
        )
    viewer.layers['CD31'].visible = False
    
    # Setup point layer
    point_layer = viewer.add_points(
        name='Points',
        size=160,
        face_color=[0]*4,
        edge_color='gray',
        edge_width=0.05,
        )
    viewer.dims.set_point(0, 0)
    
    # Functions
    def mark_points(layer, event):
        image_index = viewer.dims.current_step[0]
        pos = np.round(event.position).astype(int)
        if image_index not in points:
            points[image_index] = []
        points[image_index].append(pos[1:])
    
    def next_image():
        current_index = viewer.dims.current_step[0]
        if current_index < viewer.dims.range[0][1] - 1:
            point_layer.data = []
            viewer.dims.set_point(0, current_index + 1)
        else:
            print('End of image stack.')
            
    # Shortcuts
    @napari.Viewer.bind_key('Enter', overwrite=True)
    def next_image_key(viewer):
        next_image()
    
    @napari.Viewer.bind_key('p', overwrite=True)
    def switch_channel(viewer):
        viewer.layers['Laminin'].visible = False
        viewer.layers['CD31'].visible = True
        yield
        viewer.layers['Laminin'].visible = True
        viewer.layers['CD31'].visible = False
        
    # Setup and update viewer
    viewer.window.add_dock_widget(magicgui(next_image, call_button='Next Image (Enter)'))
    viewer.mouse_drag_callbacks.append(mark_points)
    point_layer.mode = 'add'
    napari.run()
    
    # Format and save ROI coordinates as csv
    roiCoords = []
    for i in range(len(points)):
        coords = points[i]
        for coord in coords:
            roiCoords.append(tuple((czi_paths[i].name, coord[0], coord[1])))
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(roiCoords)
        
#%% Functions -----------------------------------------------------------------

def get_image_data(czi_path, C1, C2):
    
    # Make validEDM
    validCoords = np.array(
        [coord[1:] for coord in roiCoords if coord[0] == czi_path.name]
        )
    validEDM = np.zeros_like(C1, dtype=bool)
    validEDM[validCoords[:,0], validCoords[:,1]] = True
    validEDM = distance_transform_edt(np.invert(validEDM))
    
    # Append image_data
    image_data = ((czi_path.name, C1, C2, validCoords, validEDM))

    return image_data

# -----------------------------------------------------------------------------

def get_patch_data(
        img, patchSize, patchStep, patchFreq, rSeed=0, validEDM=None
        ):
    
    # Extract variables
    img = minmax_scale(img)
    nY, nX = img.shape[0], img.shape[1]
    
    # Define patch coordinates
    xRange = np.arange(0, nX - patchSize, patchStep)
    yRange = np.arange(0, nY - patchSize, patchStep)  
    patchCoords = np.column_stack((
        np.repeat(xRange, len(yRange)),
        np.tile(yRange, len(xRange)),
        ))
    
    # Randomly remove patch coordinates
    np.random.seed(rSeed)
    nRows = int(patchFreq * patchCoords.shape[0])
    idx = np.random.choice(patchCoords.shape[0], nRows, replace=False)
    patchCoords = patchCoords[idx, :]
    
    # Extract patch
    patch_data = []
    for patchCoord in patchCoords:
        x0, y0 = patchCoord[0], patchCoord[1]
        xCtr, yCtr = x0 + patchSize // 2, y0 + patchSize // 2, 
        validDist = np.nan if validEDM is None else validEDM[yCtr, xCtr]
        patch_data.append((
            img[y0:y0 + patchSize, x0:x0 + patchSize],
            xCtr, yCtr, validDist,
            ))
            
    return patch_data

# -----------------------------------------------------------------------------

def get_generator(patch, label=None, batchSize=batchSize):    
    if label is not None:
        patch = list(zip(patch, label))
    while True:
        for i in range(0, len(patch), batchSize):
            if label is None:
                patch_batch = np.array(patch[i: i+batchSize])
                yield patch_batch
            else:
                batch = np.array(patch[i: i+batchSize])
                patch_batch, label_batch = zip(*batch)
                patch_batch = np.array(patch_batch)
                label_batch = np.array(label_batch)
                yield patch_batch, label_batch
       
#%% Define & compile neural network
       
input_shape = (patchSize, patchSize, 1)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# ---
model.add(Conv2D(32, (3, 3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# ---
model.add(Conv2D(64, (3, 3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# ---
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# ---
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
    )

#%% Process -------------------------------------------------------------------

if trainModel:

    # Get image_data ----------------------------------------------------------
    
    start = time.time()
    print('Get image_data')
    
    image_data = Parallel(n_jobs=-1)(
        delayed(get_image_data)(czi_path, C1, C2) 
        for (czi_path, C1, C2) in zip(czi_paths, C1s, C2s)
        )
        
    end = time.time()
    print(f'{(end-start):5.3f}s') 
    
    # Get patch_data ----------------------------------------------------------
    
    start = time.time()
    print('get_patch_data')
    
    patch_data = []
    for data in image_data:
        patch_data.append(get_patch_data(
            data[1], patchSize, patchStep, patchFreq, rSeed=0, validEDM=data[4]
            )) 
    patch_data = [item for data in patch_data for item in data]
    
    end = time.time()
    print(f'{(end-start):5.3f}s') 
    
    # Format datasets ---------------------------------------------------------
    
    start = time.time()
    print('Format datasets')
    
    # Split dataset (train & val)
    patch = [patch[0] for patch in patch_data]
    label = [patch[3] for patch in patch_data]
    label = (np.array(label) < maxDist).astype(int)
    patch_train, patch_val, label_train, label_val = train_test_split(
        patch, label, test_size=0.2
        )
    
    # Get generators
    generator_train = get_generator(
        patch_train, label=label_train, batchSize=batchSize
        )
    generator_val = get_generator(
        patch_val, label=label_val, batchSize=batchSize
        )
    
    end = time.time()
    print(f'{(end-start):5.3f}s') 
    
    # Train neural network ----------------------------------------------------
    
    start = time.time()
    print('Train neural network')
    
    # Train 
    steps_per_epoch = len(patch_train) // batchSize 
    validation_steps = len(patch_val) // batchSize
    history = model.fit(
        generator_train, 
        validation_data=generator_val, 
        validation_steps=validation_steps,
        steps_per_epoch=steps_per_epoch, 
        epochs=5, 
        batch_size=batchSize, 
        )
    
    # Plot results
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Save model
    model.save(Path(model_path, 'model.h5')) 
    
    end = time.time()
    print(f'{(end-start):5.3f}s') 

#%% Predict -------------------------------------------------------------------

from scipy.interpolate import griddata
from skimage.feature import peak_local_max

# Load model
model = load_model(Path(model_path, 'model.h5'))

# Format data
C1 = minmax_scale(C1s[3])

# Get patch_data
patch_data = get_patch_data(
    C1, patchSize, patchStep, 1, rSeed=0, validEDM=None)
patch = [patch[0] for patch in patch_data]

# Get generators
generator = get_generator(
    patch, label=None, batchSize=batchSize
    )

# Get predictions
steps = len(patch) // batchSize + (len(patch) % batchSize != 0)
predictions = model.predict(generator, steps=steps)

# Generate prediction map (pMap)
pMap = np.zeros_like(C1) * np.nan
xCtr = [data[1] for data in patch_data]
yCtr = [data[2] for data in patch_data]
for i, (x, y) in enumerate(zip(xCtr, yCtr)):
    pMap[y, x] = predictions[i]
y, x = np.mgrid[0:pMap.shape[0], 0:pMap.shape[1]]
mask = ~np.isnan(pMap)
pMap = griddata((y[mask], x[mask]), pMap[mask], (y, x), method='cubic')

# 
coords = peak_local_max(
    pMap, min_distance=patchSize, threshold_abs=0.2
    ).astype(int)
    
# Display
viewer = napari.Viewer()
viewer.add_image(C1)
viewer.add_image(pMap)
points_layer = viewer.add_points(
    coords, 
    size=patchSize,
    edge_width=0.1,
    edge_color='red',
    face_color='transparent',
    opacity = 0.5,
    )
