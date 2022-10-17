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

### 2.3 Label Texts
load images 
webserver (create_text_labels_editor.py)
json per image with key value pairs like that:
```json
{
    "text": "darig", 
    "background": "concrete", 
    "fill": "white", 
    "outline": "red, black", 
    "style": "block", 
    "highlights": ""
}
```

### 2.4 create free text descriptions from jsons

"a graffiti reading 'XXXX' on a 'background', with 'fill' fill-color, 'outline' outline(s), in a 'style' style. with 'highlights'" 
and variantions thereof, thereby creating a larger dataset


### 2.3 Possible Options

# continue here!

    python3 editor.py -i ~/Pictures/GANfitti/raw_graff_data/ -o ~/Pictures/GANfitti/labels`

    use yolov5 to test if labeling can already be automated


# label texts
    
    python3 create_text_labels_editor.py -i ~/Pictures/GANfitti/generated_512_v3/ -o ~/Pictures/GANfitti/text_labels/