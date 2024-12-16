![Python Badge](https://img.shields.io/badge/Python-3.10-rgb(69%2C132%2C182)?logo=python&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))  
![Author Badge](https://img.shields.io/badge/Author-Benoit%20Dehapiot-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))
![Date Badge](https://img.shields.io/badge/Created-2023--06--27-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))
![License Badge](https://img.shields.io/badge/Licence-GNU%20General%20Public%20License%20v3.0-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))     

# ETH-ScopeM_Masschelein  
Muscular blood vessel analysis tool

## Index
- [Installation](#installation)
- [Usage](#usage)
- [Comments](#comments)
- [Comments](#comments)

## Installation

Pease select your operating system

<details> <summary>Windows</summary>  

### Step 1: Download this GitHub Repository 
- Click on the green `<> Code` button and download `ZIP` 
- Unzip the downloaded file to a desired location

### Step 2: Install Miniforge (Minimal Conda installer)
- Download and install [Miniforge](https://github.com/conda-forge/miniforge) for your operating system   
- Run the downloaded `.exe` file  
    - Select "Add Miniforge3 to PATH environment variable"  

### Step 3: Setup Conda 
- Open the newly installed Miniforge Prompt  
- Move to the downloaded GitHub repository
- Run the following command:  
```bash
mamba env create -f environment.yml
```
- Activate Conda environment:
```bash
conda activate Masschelein
```
Your prompt should now start with `(Masschelein)` instead of `(base)`

</details> 

<details> <summary>MacOS</summary>  

### Step 1: Download this GitHub Repository 
- Click on the green `<> Code` button and download `ZIP` 
- Unzip the downloaded file to a desired location

### Step 2: Install Miniforge (Minimal Conda installer)
- Download and install [Miniforge](https://github.com/conda-forge/miniforge) for your operating system   
- Open your terminal
- Move to the directory containing the Miniforge installer
- Run one of the following command:  
```bash
# Intel-Series
bash Miniforge3-MacOSX-x86_64.sh
# M-Series
bash Miniforge3-MacOSX-arm64.sh
```   

### Step 3: Setup Conda 
- Re-open your terminal 
- Move to the downloaded GitHub repository
- Run the following command: 
```bash
mamba env create -f environment.yml
```  
- Activate Conda environment:  
```bash
conda activate Masschelein
```
Your prompt should now start with `(Masschelein)` instead of `(base)`

</details>

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

## Comments 
### Meeting 16/12/2024
- Export and display values per patient (also for statistics).
- Thickness as width at half prominence.

## Comments