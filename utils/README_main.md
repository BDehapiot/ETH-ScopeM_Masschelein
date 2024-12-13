## Usage

### `select.py`
Opens all the `czi` images in the `data_path` folder and displays them in a 
custom Napari window. Click on the desired objects and move on to the next 
image. The coordinates of selected objects are saved as `coords.csv` in the 
`data_path` folder.

```bash
# Paths
- data_path              # str, path to folder containing image(s) to process
```
```bash
# Parameters
- subfolders             # bool, process or not "data_path" subfolders
- C1_contrast_limits     # list, min & max C1
- C2_contrast_limits     # list, min & max C2
```
```bash
# Outputs
- coords.csv             # coordinates of selected objects (img, y, x)
```

### `analyse.py`
Read `coords.csv` and analyse selected objects by splitting data between
condition #1 (control) and condition #2 (PAD). The code outputs : averaged 
cropped images `..._crops_..._avg.tif`, averaged radial fluo. int. profiles
`..._radInts.csv`, and the distance of the max. radial fluo. int. profiles 
`..._radInts_max.csv` with statistics.

```bash
# Paths
- data_path              # str, path to folder containing image(s) to process
```
```bash
# Parameters
- crop_size              # int, object crop window size (pixels)
- ctr                    # list, control patient ids 
- pad                    # list, pad patient ids
```
```bash
# Outputs
- C1_crops_cond1_avg.tif # avg. crop images C1 - cond1
- C1_crops_cond2_avg.tif # avg. crop images C1 - cond2
- C2_crops_cond1_avg.tif # avg. crop images C2 - cond1
- C2_crops_cond2_avg.tif # avg. crop images C2 - cond2
- C1_radInts.csv         # avg. & std. radial fluo. int. profiles C1 - cond1 & cond2
- C2_radInts.csv         # avg. & std. radial fluo. int. profiles C2 - cond1 & cond2
- C1_radInts_max.csv     # dist. of max. radial fluo. int. C1 - cond1 & cond2 + stats
- C2_radInts_max.csv     # dist. of max. radial fluo. int. C2 - cond1 & cond2 + stats
```