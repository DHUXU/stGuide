# stGuide advances label transfer in spatial transcriptomics through attention-based supervised graph representation learning
## Overview
![图片1](https://github.com/user-attachments/assets/f8aeee7b-ebe9-4116-b9a6-fd9617f9d6e4)
**Overview of stGuide.** a Given multiple ST datasets with three-layer profiles: gene expression in both query and reference, spatial locations, and annotations in the reference data, stGuide annotates query spot labels based on the reference annotations. A composed graph is constructed using -nearest neighbors (KNN) for spatial locations and mutual nearest neighbors (MNN) for transcriptomics within and across slices. b stGuide extracts reference spot representations by aggregating information from spatially and transcriptomic similar spots, guided by spot annotations. c stGuide learns shared representations for the reference and query datasets by aggregating messages from spatially and transcriptomic similar spots, aligning  with the feature space of  to enable label transfer. d The representations of query and reference datasets are used for label transfer and pseudo-time analysis.
## Installation
The installation was tested on a machine with a 40-core Intel(R) Xeon(R) Silver 4210R CPU, 128GB of RAM, and an NVIDIA A800 GPU with 32GB of RAM, using Python 3.8.17. If possible, please run stGuide on CUDA.
### 1.Grab source code of stGuide
```bash
git clone https://github.com/YupengXu1/stGuide.git
cd stGuide
```
### 2. Install stGuide in the virtual environment by conda
* Firstly, install conda: https://docs.anaconda.com/anaconda/install/index.html
* Then, automatically install all used packages (described by "requirements.yml") for stGuide in a few mins.

```bash
conda config --set channel_priority strict
conda env create -f requirements.yml
conda activate stGuide
```
### 3. Install R packages
Our MNN program is deployed on R software and rpy2 library, please install r-base and related package dependecies via conda.
Run the following commands in Linux Bash Shell:

```bash
conda install r-base
conda install r-dplyr (here, magrittr, tidyverse, batchelor, BiocParallel, FNN)
```
Or you can install these package dependencies by install.packages() and BiocManager::install() commands in R script.
Note: To reduce your waiting time, we recommend using the rpy2 library to call the path of R software installed in your existing virtual environment.

## Quick start


