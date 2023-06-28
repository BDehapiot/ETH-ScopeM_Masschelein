#%% Imports -------------------------------------------------------------------

import csv
import napari
import numpy as np
from pathlib import Path
from magicgui import magicgui
from pylibCZIrw import czi as pyczi
from czitools import extract_metadata

#%% Comments ------------------------------------------------------------------

''' C1 = laminin (cell membrane marker) '''
''' C2 = CD31 (endothelial cell marker) '''

#%% Parameters ----------------------------------------------------------------

selectROIs = False
patchSize = 160
patchStep = 10

#%% Initialize ----------------------------------------------------------------

data_path = Path('D:\local_Masschelein\data')
model_path = Path('D:\local_Masschelein\data\model')

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
    with open(str(model_path) + '/roiCoords.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(roiCoords)
        
#%%

#%% Extract ROIs --------------------------------------------------------------

# patch_data = []
# for czi_path in czi_paths:
    
#     # Extract metadata
#     metadata = extract_metadata(str(czi_path))
    
#     # Define patch coordinates
#     nX, nY = metadata['nX'], metadata['nY']
#     xRange = np.arange(0, nX - patchSize, patchStep)
#     yRange = np.arange(0, nY - patchSize, patchStep)  
#     coords = np.column_stack((
#         np.repeat(xRange, len(yRange)),
#         np.tile(yRange, len(xRange)),
#         ))
    
#     # Open images
#     with pyczi.open_czi(str(czi_path)) as czidoc:
#         C1 = czidoc.read(plane={'T': 0, 'Z': 0, 'C': 0}).squeeze()
#         C2 = czidoc.read(plane={'T': 0, 'Z': 0, 'C': 1}).squeeze()
        
#     # Create patches
#     for coord in coords:
#         x0, y0 = coord[0], coord[1]
#         xMid, yMid = x0 + patchSize // 2, y0 + patchSize // 2, 
#         patch_data.append((
#             czi_path.name, xMid, yMid,
#             C1[y0:y0 + patchSize, x0:x0 + patchSize],
#             C2[y0:y0 + patchSize, x0:x0 + patchSize],
#             ))
        
#%%

# name = '4.1.czi'
# x = 550
# y = 377

# names = np.array([data[0] for data in patch_data])
# xMid = np.array([data[1] for data in patch_data])
# yMid = np.array([data[2] for data in patch_data])
# coords = np.array(list(zip(xMid, yMid)))
# dist = np.sum((coords - (x, y)) ** 2, axis=1)
# mask = (names == name)
# idxs = np.where(mask)[0]
# idx = idxs[np.argmin(dist[mask])]

# img = patch_data[idx][3]

# import napari
# viewer = napari.Viewer()
# viewer.add_image(img)
   
#%% Save data -----------------------------------------------------------------
    