#%% Imports -------------------------------------------------------------------

import csv
import napari
import numpy as np
from pathlib import Path
from magicgui import magicgui
import matplotlib.pyplot as plt
from pylibCZIrw import czi as pyczi
from scipy.ndimage import distance_transform_edt
from sklearn.preprocessing import minmax_scale

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#%% Comments ------------------------------------------------------------------

''' C1 = laminin (cell membrane marker) '''
''' C2 = CD31 (endothelial cell marker) '''

#%% Parameters ----------------------------------------------------------------

selectROIs = False
patchSize = 160
patchStep = 10

#%% Initialize ----------------------------------------------------------------

data_path = Path('D:/local_Masschelein/data')
model_path = Path('D:/local_Masschelein/data/model')
csv_path = Path('D:/local_Masschelein/data/model/roiCoords.csv')

# Get paths & open images
img_data = []
for czi_path in data_path.iterdir():
    if czi_path.suffix == '.czi':
        with pyczi.open_czi(str(czi_path)) as czidoc:
            img_data.append((
                czi_path.name,
                czidoc.read(plane={'T': 0, 'Z': 0, 'C': 0}).squeeze(),
                czidoc.read(plane={'T': 0, 'Z': 0, 'C': 1}).squeeze(),
                ))
            
# Open csv file
if not selectROIs:
    with open(str(csv_path), 'r') as f:
        reader = csv.reader(f)
        roiCoords = [tuple(map(int, row)) for row in reader]
            
#%% Select ROIs ---------------------------------------------------------------

if selectROIs:
    
    # Initialize
    points = {}
    viewer = napari.Viewer()
    C1 = np.stack([data[1] for data in img_data])
    C2 = np.stack([data[2] for data in img_data])
    hstack = np.stack((C1, C2), axis=0)

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
            roiCoords.append(tuple((i, coord[0], coord[1])))
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(roiCoords)
        
#%% Make patches --------------------------------------------------------------

patch_data = []
for i, data in enumerate(img_data):
    
    # Extract variables
    C1, C2 = minmax_scale(data[1]), minmax_scale(data[2])
    nY, nX = C1.shape[0], C1.shape[1]
    
    # Extract roiCoords and make EDM
    coords = np.array([coord[1:] for coord in roiCoords if coord[0] == i])
    EDM = np.zeros_like(C1, dtype=bool)
    EDM[coords[:,0], coords[:,1]] = True
    EDM = distance_transform_edt(np.invert(EDM))
    
    # Define patch coordinates
    xRange = np.arange(0, nX - patchSize, patchStep)
    yRange = np.arange(0, nY - patchSize, patchStep)  
    patchCoords = np.column_stack((
        np.repeat(xRange, len(yRange)),
        np.tile(yRange, len(xRange)),
        ))

    # Create patches
    for patchCoord in patchCoords:
        x0, y0 = patchCoord[0], patchCoord[1]
        xCtr, yCtr = x0 + patchSize // 2, y0 + patchSize // 2, 
        patch_data.append((
            czi_path.name, xCtr, yCtr, EDM[yCtr, xCtr],
            C1[y0:y0 + patchSize, x0:x0 + patchSize],
            C2[y0:y0 + patchSize, x0:x0 + patchSize],
            ))

#%% Train ---------------------------------------------------------------------

import random
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split

maxDist = 20
subSize = 100000
random.seed(0)

# Extract & format data for training
patches = [data[4] for data in patch_data]
classes = np.array([data[3] for data in patch_data])
classes = (classes < maxDist).astype(int)
subIdx = random.sample(range(len(patches)), subSize)
patches_train = np.stack([patches[i] for i in subIdx])
classes_train = np.stack([classes[i] for i in subIdx])
X_train, X_test, y_train, y_test = train_test_split(
    patches_train, 
    classes_train, 
    test_size = 0.20, 
    random_state = 0
    )

# Define & compile neural network
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

# Train network
history = model.fit(
    X_train, y_train, 
    batch_size = 64, 
    verbose = 1, 
    epochs = 10,      
    validation_data=(X_test,y_test),
    shuffle = False
    )

# Save model
model.save(Path(model_path, 'model.h5')) 

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

acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#%% Predict -------------------------------------------------------------------

random.seed(1)
subSize = 10000
subIdx = random.sample(range(len(patches)), subSize)
patches_predict = np.stack([patches[i] for i in subIdx])
classes_predict = model.predict(patches_predict)

validC1 = np.array([data for i, data in enumerate(patches_predict) if classes_predict[i] > 0.0005])
viewer = napari.Viewer()
viewer.add_image(validC1)

#%% Check valid patches -------------------------------------------------------

# maxDist = 20
# validC1 = np.stack([data[4] for data in patch_data if data[3] < maxDist])
# viewer = napari.Viewer()
# viewer.add_image(validC1)
   
#%% Save data -----------------------------------------------------------------
    