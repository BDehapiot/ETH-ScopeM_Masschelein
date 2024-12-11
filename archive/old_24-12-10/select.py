#%% Info ----------------------------------------------------------------------

#%% Imports -------------------------------------------------------------------

import csv
import napari
import numpy as np
from pathlib import Path
from magicgui import magicgui
from pylibCZIrw import czi as pyczi

#%% Initialize ----------------------------------------------------------------

data_path = Path('D:/local_Masschelein/data')
model_path = Path('D:/local_Masschelein/data/model')
csv_path = Path('D:/local_Masschelein/data/model/valid_coords.csv')

# Get paths
czi_paths = []
for path in data_path.iterdir():
    if path.suffix == '.czi':
        czi_paths.append(path)

# Open images
C1s, C2s = [], []
for czi_path in czi_paths:      
    with pyczi.open_czi(str(czi_path)) as czidoc:
        C1s.append(czidoc.read(plane={'T': 0, 'Z': 0, 'C': 0}).squeeze())
        C2s.append(czidoc.read(plane={'T': 0, 'Z': 0, 'C': 1}).squeeze())
        
#%% Annotate ------------------------------------------------------------------

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
        viewer.window._qt_window.close()
        print('End of image stack.')
        
def save_csv_on_close(event):
    valid_coords = []
    for i in points:
        coords = points[i]
        for coord in coords:
            valid_coords.append(tuple((czi_paths[i].name, coord[0], coord[1])))
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(valid_coords)
    print('CSV file saved on window close.')
        
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
viewer.window._qt_window.closeEvent = save_csv_on_close
viewer.mouse_drag_callbacks.append(mark_points)
point_layer.mode = 'add'
napari.run()