#%% Imports -------------------------------------------------------------------

import numpy as np
from pathlib import Path
from pylibCZIrw import czi as pyczi
# from joblib import Parallel, delayed 

# Napari
import napari
from napari.layers.labels.labels import Labels
from napari.layers.points.points import Points

# Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QPushButton, QGroupBox, QVBoxLayout, QWidget, QLabel)

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path('D:/local_Masschelein/data')

#%% Function(s) ---------------------------------------------------------------

def open_czi(path):
    with pyczi.open_czi(str(path)) as czidoc:
        C1 = czidoc.read(plane={'T': 0, 'Z': 0, 'C': 0}).squeeze()
        C2 = czidoc.read(plane={'T': 0, 'Z': 0, 'C': 1}).squeeze()
    return C1, C2

#%% Class : Select() ----------------------------------------------------------

class Select:

    def __init__(self, paths):
        self.paths = paths
        self.idx = 0
        self.open_czi()
        self.init_points()
        self.init_viewer()
        self.get_info_text() 
             
    # Open
    def open_czi(self):
        self.C1s, self.C2s = [], []
        for path in self.paths:
            C1, C2 = open_czi(path)
            self.C1s.append(C1)
            self.C2s.append(C2)
        
    # Points
    def add_points(self):
        points = self.viewer.layers["points"].data
        if points.shape[0] > 0:
            self.points[self.idx] = points
        
    def save_points(self):
        csv = []
        for i, points in enumerate(self.points):
            if points.shape[0] > 0:
                idx = np.full((points.shape[0], 1), i, dtype=float)
                csv.append(np.hstack((idx, points)))
        csv = np.concatenate(csv).astype(int)
        np.savetxt("coords.csv", csv, delimiter=",", fmt="%d")
        
    def load_points(self):
        csv = np.loadtxt("coords.csv", delimiter=",").astype(int)
        for i, points in enumerate(csv):
            self.points[points[0]] = np.append(
                self.points[points[0]], values=[[points[1], points[2]]], axis=0) 
    
    def init_points(self):
        self.points = [np.empty((0, 2)) for _ in range(len(self.paths) - 1)] 
        if Path("coords.csv").is_file():
            self.load_points()       

#%% Viewer --------------------------------------------------------------------
        
    def init_viewer(self):
        
        # Setup viewer
        self.viewer = napari.Viewer()
        self.viewer.add_image(self.C2s[self.idx], name="C2")
        self.viewer.add_image(self.C1s[self.idx], name="C1")
        self.viewer.add_points(
            name='points', size=120, edge_width=0.05,
            face_color=[0]*4, edge_color='gray',
            )
        self.viewer.layers["points"].data = self.points[self.idx] 
        self.viewer.layers["points"].mode = "add"
        self.viewer.layers["points"].opacity = 0.5
        
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
            
        @Points.bind_key("Backspace", overwrite=True)
        def clear_image_points_key(viewer):
            self.clear_image_points()
            
        @Points.bind_key("Ctrl-Backspace", overwrite=True)
        def clear_all_points_key(viewer):
            self.clear_all_points()
            
    #%% Function(s) -----------------------------------------------------------
        
    # Shortcuts
    
    def prev_image(self):
        if self.idx > 0:
            self.add_points()
            self.save_points()
            self.idx -= 1
            self.display_images()

    def next_image(self):
        if self.idx < len(self.paths) - 1:
            self.add_points()
            self.save_points()
            self.idx += 1
            self.display_images()
            
    def clear_image_points(self):
        self.points[self.idx] = []
        self.display_images()
        
    def clear_all_points(self):
        self.init_points()
        self.display_images()
            
    # Procedures
            
    def display_images(self):
        self.viewer.layers["C1"].data = self.C1s[self.idx].copy()
        self.viewer.layers["C2"].data = self.C2s[self.idx].copy()
        self.viewer.layers["points"].data = self.points[self.idx] 
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
            f"<p{style0}>Image<br><br>"
            f"<span{style1}>{img_name}</span><br>"
            )
        
        self.info_short.setText(
            f"<p{style0}>Shortcuts<br><br>" 
            
            f"<span{style2}>- Next/Prev Image {spacer * 0}:</span>"
            f"<span{style3}> Page[Up/Down]</span><br>"
            
            f"<span{style2}>- Edit Outline    {spacer * 3}:</span>"
            f"<span{style3}> Shift</span><br>" 
            
            f"<span{style2}>- Save Outline    {spacer * 3}:</span>"
            f"<span{style3}> Enter</span><br>"  
            
            f"<span{style2}>- Revert Outline  {spacer * 1}:</span>"
            f"<span{style3}> Delete</span><br>"
            
            f"<span{style2}>- Hide Outline    {spacer * 3}:</span>"
            f"<span{style3}> Backspace</span><br>"  
            
            f"<span{style2}>- Pan Image       {spacer * 6}:</span>"
            f"<span{style3}> Space or Num[0]</span><br>" 
            
            )    
        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Get paths
    czi_paths = list(data_path.glob("*.czi"))

    # Select
    select = Select(czi_paths)
    points = select.points
    
    # csv = []
    # for i, pts in enumerate(points):
    #     if pts.shape[0] > 0:
    #         idx = np.full((pts.shape[0], 1), i, dtype=float)
    #         csv.append(np.hstack((idx, pts)))
    # csv = np.concatenate(csv).astype(int)
    # np.savetxt("coords.csv", csv, delimiter=",", fmt="%d")
    
    # csv_new = np.loadtxt("coords.csv", delimiter=",").astype(int)
    
    # points_new = [np.empty((0, 2)) for _ in range(len(czi_paths) - 1)]
    # for i, pts in enumerate(csv_new):
    #     points_new[pts[0]] = np.append(
    #         points_new[pts[0]], values=[[pts[1], pts[2]]], axis=0) 
        
    
    pass