#%% Info ----------------------------------------------------------------------

#%% Imports -------------------------------------------------------------------

import csv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
    )

from utils import ImageData, get_patch, get_generator

#%% Parameters ----------------------------------------------------------------

patchSize = 80
patchStep = 10
patchFreq = 0.1
rSeed = 0
maxDist = 20
batchSize = 128

#%% Initialize ----------------------------------------------------------------

data_path = Path('D:/local_Masschelein/data')
model_path = Path('D:/local_Masschelein/data/model')
csv_path = Path('D:/local_Masschelein/data/model/valid_coords.csv')

# Get paths
czi_paths = []
for path in data_path.iterdir():
    if path.suffix == '.czi':
        czi_paths.append(path)
        
# Open csv file
with open(str(csv_path), 'r') as f:
    reader = csv.reader(f)
    valid_coords = [
        tuple(int(x) if x.isdigit() else x for x in row) for row in reader
        ]
    
#%% Neural network ------------------------------------------------------------
       
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

#%% Format data ---------------------------------------------------------------

# Get image_data
image_data = Parallel(n_jobs=-1)(
    delayed(ImageData)(path, annotations=valid_coords) for path in czi_paths
    )

# Get patch_data
patch_data = [
    get_patch(image, patchSize, patchStep, patchFreq)
    for image in image_data
    ]
patch_data = [item for data in patch_data for item in data]

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

#%% Train model ---------------------------------------------------------------

# Train neural network
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