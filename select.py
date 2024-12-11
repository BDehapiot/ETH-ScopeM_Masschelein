#%% Imports -------------------------------------------------------------------

import numpy as np
from pathlib import Path

# Napari
import napari
from napari.layers.points.points import Points

# Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QPushButton, QGroupBox, QVBoxLayout, QWidget, QLabel)

# Functions
from functions import open_czi

#%% Comments ------------------------------------------------------------------

'''
- Maybe find a way to match image name with coords (if we add data in the folder)
- Add point deletion shortcut
'''

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path('D:/local_Masschelein/data')

#%% Class : Select() ----------------------------------------------------------

class Select:

    def __init__(self, data_path, subfolders=False):
        self.data_path = data_path
        self.idx = 0
            
        # Paths
        if subfolders: 
            self.paths = list(self.data_path.rglob("*.czi"))
        else:
            self.paths = list(self.data_path.glob("*.czi"))
        self.coords_path = Path(data_path, "coords.csv")
        
        # Run
        self.open_czi()
        self.init_coords()
        self.init_viewer()
        self.get_info_text() 
        
    # Open
    def open_czi(self):
        self.C1s, self.C2s = open_czi(self.paths)
        
    # coords
    def add_coords(self):
        coords = self.viewer.layers["coords"].data
        if coords.shape[0] > 0:
            self.coords[self.idx] = coords
        
    def save_coords(self):
        coords_csv = []
        for i, img_coords in enumerate(self.coords):
            if img_coords.shape[0] > 0:
                idx = np.full((img_coords.shape[0], 1), i, dtype=float)
                coords_csv.append(np.hstack((idx, img_coords)))
            else:
                coords_csv.append(np.empty((0, 3)))
        coords_csv = np.concatenate(coords_csv).astype(int)
        np.savetxt(self.coords_path, coords_csv, delimiter=",", fmt="%d")
            
    def init_coords(self, empty=False):
        self.coords = [np.empty((0, 2)) for _ in range(len(self.paths) - 1)] 
        if Path(self.coords_path).is_file() and not empty:
            coords_csv = np.loadtxt(self.coords_path, delimiter=",").astype(int)
            if coords_csv.ndim == 1 and coords_csv:
                coord = coords_csv
                self.coords[coord[0]] = np.append(
                    self.coords[coord[0]], values=[[coord[1], coord[2]]], axis=0) 
            elif coords_csv.ndim == 2:
                for i, coord in enumerate(coords_csv):
                    self.coords[coord[0]] = np.append(
                        self.coords[coord[0]], values=[[coord[1], coord[2]]], axis=0) 

#%% Viewer --------------------------------------------------------------------
        
    def init_viewer(self):
        
        # Setup viewer
        self.viewer = napari.Viewer()
        self.viewer.add_image(self.C2s[self.idx], name="C2", colormap="magenta")
        self.viewer.add_image(self.C1s[self.idx], name="C1", colormap="green")
        self.viewer.add_points(
            name='coords', size=120, border_width=0.05,
            face_color=[0]*4, border_color="grey",
            )
        self.viewer.layers["coords"].data = self.coords[self.idx] 
        self.viewer.layers["coords"].mode = "add"
        # self.viewer.layers["coords"].opacity = 0.5
        
        # Create "Actions" menu
        self.act_group_box = QGroupBox("Actions")
        act_group_layout = QVBoxLayout()
        self.btn_next_image = QPushButton("Next Image")
        self.btn_prev_image = QPushButton("Previous Image")
        act_group_layout.addWidget(self.btn_next_image)
        act_group_layout.addWidget(self.btn_prev_image)
        self.act_group_box.setLayout(act_group_layout)
        self.btn_next_image.clicked.connect(self.next_image)
        self.btn_prev_image.clicked.connect(self.prev_image)
        
        # Create text
        self.info_image = QLabel()
        self.info_image.setFont(QFont("Consolas"))
        self.info_short = QLabel()
        self.info_short.setFont(QFont("Consolas"))
        
        # Create layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.act_group_box)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.info_image)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.info_short)
        
        # Create widget
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.viewer.window.add_dock_widget(
            self.widget, area="right", name="Correct") 
        
        # Shortcuts

        @self.viewer.bind_key("PageDown", overwrite=True)
        def previous_image_key(viewer):
            self.prev_image()
        
        @self.viewer.bind_key("PageUp", overwrite=True)
        def next_image_key(viewer):
            self.next_image()
            
        @self.viewer.bind_key("Enter", overwrite=True)
        def toggle_C2_key(viewer):
            self.viewer.layers["C1"].visible = False
            yield
            self.viewer.layers["C1"].visible = True
            
        @self.viewer.bind_key("0", overwrite=True)
        def pan_switch_key0(viewer):
            self.viewer.layers["coords"].mode = "pan_zoom"
            yield
            self.viewer.layers["coords"].mode = "add"
        
        @self.viewer.bind_key("Space", overwrite=True)
        def pan_switch_key1(viewer):
            self.viewer.layers["coords"].mode = "pan_zoom"
            yield
            self.viewer.layers["coords"].mode = "add"
            
        @Points.bind_key("Backspace", overwrite=True)
        def clear_image_coords_key(viewer):
            self.clear_image_coords()
            
        @Points.bind_key("Ctrl-Backspace", overwrite=True)
        def clear_all_coords_key(viewer):
            self.clear_all_coords()
            
    #%% Function(s) -----------------------------------------------------------
        
    # Shortcuts
    
    def prev_image(self):
        if self.idx > 0:
            self.add_coords()
            self.save_coords()
            self.idx -= 1
            self.display_images()

    def next_image(self):
        if self.idx < len(self.paths) - 1:
            self.add_coords()
            self.save_coords()
            self.idx += 1
            self.display_images()
                        
    def clear_image_coords(self):
        self.coords[self.idx] = np.empty((0, 2))
        self.save_coords()
        self.display_images()
        
    def clear_all_coords(self):
        self.init_coords(empty=True)
        self.save_coords()
        self.display_images()
                    
    # Procedures
            
    def display_images(self):
        self.viewer.layers["C1"].data = self.C1s[self.idx].copy()
        self.viewer.layers["C2"].data = self.C2s[self.idx].copy()
        self.viewer.layers["coords"].data = self.coords[self.idx] 
        self.get_info_text() 
        
    # Text 
    
    def get_info_text(self):
                           
        def set_style(color, size, weight, decoration):
            return (
                " style='"
                f"color: {color};"
                f"font-size: {size}px;"
                f"font-weight: {weight};"
                f"text-decoration: {decoration};"
                "'"
                )

        img_name = self.paths[self.idx].name

        font_size = 12
        # Set styles (Titles)
        style0 = set_style("White", font_size, "normal", "underline")
        # Set styles (Filenames)
        style1 = set_style("Khaki", font_size, "normal", "none")
        # Set styles (Legend)
        style2 = set_style("LightGray", font_size, "normal", "none")
        # Set styles (Shortcuts)
        style3 = set_style("LightSteelBlue", font_size, "normal", "none")
        spacer = "&nbsp;"

        self.info_image.setText(
            f"<p{style0}>Image ({self.idx}/{len(self.C1s)})<br><br>"
            f"<span{style1}>{img_name}</span><br>"
            )
        
        self.info_short.setText(
            f"<p{style0}>Shortcuts<br><br>" 
            
            f"<span{style2}>- Next/Prev Image   {spacer * 2}:</span>"
            f"<span{style3}> Page[Up/Down]</span><br>"
            
            f"<span{style2}>- Show C2           {spacer * 10}:</span>"
            f"<span{style3}> Enter</span><br>" 
            
            f"<span{style2}>- Pan img           {spacer * 10}:</span>"
            f"<span{style3}> Space or Num[0]</span><br>" 
            
            f"<span{style2}>- Clear img. coords {spacer * 0}:</span>"
            f"<span{style3}> Backspace</span><br>"  
            
            f"<span{style2}>- Clear all coords  {spacer * 1}:</span>"
            f"<span{style3}> Ctrl + Backspace</span><br>"  
                                  
            )    
        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    select = Select(data_path, subfolders=False)