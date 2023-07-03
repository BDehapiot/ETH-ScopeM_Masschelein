#%% Info ----------------------------------------------------------------------

#%% Imports -------------------------------------------------------------------

import numpy as np
from pylibCZIrw import czi as pyczi
from sklearn.preprocessing import minmax_scale
from scipy.ndimage import distance_transform_edt

#%% Class & functions ---------------------------------------------------------

class ImageData:
    def __init__(self, path, annotations=None):
        self.path = path
        self.annotations = annotations 
        self.image = self.load_image()
        self.shape = self.get_shape()
        if annotations is not None:
            self.edm = self.get_edm()
        
    def load_image(self):            
        with pyczi.open_czi(str(self.path)) as czidoc:
            image = czidoc.read(plane={'T': 0, 'Z': 0, 'C': 0}).squeeze()
            image = minmax_scale(image)
            return image
    
    def get_shape(self):
        return (self.image.shape[0], self.image.shape[1])     
    
    def get_edm(self):  
        
        # Extract relevant coordinates from valid_coords.csv
        coords = np.array([coord[1:] for coord in self.annotations 
                           if coord[0] == self.path.name])
        
        # Compute EDM from coords
        edm = np.zeros_like(self.image, dtype=bool)
        edm[coords[:,0], coords[:,1]] = True
        edm = distance_transform_edt(np.invert(edm))
        
        return edm 
    
# -----------------------------------------------------------------------------
    
def get_patch(image, patchSize, patchStep, patchFreq, rSeed=0):
    
    # Define patch coordinates
    xRange = np.arange(0, image.shape[1] - patchSize, patchStep)
    yRange = np.arange(0, image.shape[0] - patchSize, patchStep)  
    patch_coords = np.column_stack((
        np.repeat(xRange, len(yRange)),
        np.tile(yRange, len(xRange)),
        ))
    
    # Randomly remove patch coordinates
    np.random.seed(rSeed)
    nRows = int(patchFreq * patch_coords.shape[0])
    idx = np.random.choice(patch_coords.shape[0], nRows, replace=False)
    patch_coords = patch_coords[idx, :]
    
    # Extract patch
    patch = []
    for patch_coord in patch_coords:
        x0, y0 = patch_coord[0], patch_coord[1]
        xCtr, yCtr = x0 + patchSize // 2, y0 + patchSize // 2, 
        dist = image.edm[yCtr, xCtr] if hasattr(image, 'edm') else np.nan
        patch.append((
            image.image[y0:y0 + patchSize, x0:x0 + patchSize],
            xCtr, yCtr, dist,
            ))
    
    return patch
    
# -----------------------------------------------------------------------------
    
def get_generator(patch, label=None, batchSize=64):    
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