# GANfitti

## Overview

- 1 Preface
- 2 Dataset
- 2.1 Collection
- 2.2 Cropping
- 2.3 Possible Options

## 1. Preface
tools for work on the GANfitti dataset

## 2. Dataset

### 2.1 Collection
Berlin, Kassel Spring 2022


### 2.2 Cropping
create AOI file -> one line per graffitti -> relative coords
script creates images from labels
use padd_all_images_512.sh -> or equvalent script with changed size 256 1024 etc

### 2.3 Possible Options

# continue here!

    python3 editor.py -i ~/Pictures/GANfitti/raw_graff_data/ -o ~/Pictures/GANfitti/labels`

    use yolov5 to test if labeling can already be automated
