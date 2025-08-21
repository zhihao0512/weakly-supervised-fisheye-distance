# weakly-supervised-fisheye-distance
This is the official repository of paper "Weakly Supervised Monocular Fisheye Camera Distance Estimation with Segmentation Constraints". Electronics, 2025. This code is based on [WoodScape](https://github.com/valeoai/WoodScape/tree/master), with modifications.
## Methodology
![pipeline](https://github.com/user-attachments/assets/8ed8ba2c-97f2-454b-9cbd-a062589f9110)
## Dataset preparation
1. Download the [WoodScape](https://woodscape.valeo.com/woodscape/) dataset and place it in the main directory.
2. Run ./data/WoodScape/generate_instance.py to generate semantic and instance labels.
3. Use the [FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official) code and refer to ./data/SynWoodScape/inverse_flow.py to generate optical flow labels for the WoodScape dataset.
4. Download the [SynWoodScape](https://drive.google.com/drive/folders/1N5rrySiw1uh9kLeBuOblMbXJ09YsqO7I) dataset.
5. Run the scripts in ./data/SynWoodScape to supplement the data into the WoodScape dataset.
## Data organization
```
WoodScape_ICCV19
└───rgb_images
│   │   00000_[CAM].png
│   │   00001_[CAM].png
|   |   ...
│   │   10223_[CAM].png
└───previous_images
│   │   00000_[CAM]_prev.png
│   │   00001_[CAM]_prev.png
|   |   ...
│   │   10223_[CAM]_prev.png
└───segment_annotations
        │   instance
        │   │   00000_[CAM].png
        │   │   00001_[CAM].png
        |   |   ...
        │   semantic
        │   │   00000_[CAM].png
        │   │   00001_[CAM].png
        |   |   ...
│   │
└───vehicle_data
        │   previous_images
        │   │   00000_[CAM].png
        │   │   00001_[CAM].png
        |   |   ...
        │   rgb_images
        │   │   00000_[CAM].png
        │   │   00001_[CAM].png
        |   |   ...
│   │
└───flow_annotations_flowformersintel
        │   gtLabels
        │   │   00000_[CAM].flo
        │   │   00001_[CAM].flo
        |   |   ...
        │   rgbLabels
        │   │   00000_[CAM].png
        │   │   00001_[CAM].png
        |   |   ...
│   │
└───calibration_data
│   │   00000_[CAM].json
│   │   00001_[CAM].json
|   |   ...
│   │
└───masks
│   │
└───raw_data
│   │   08234_FV.npy
|   |   ...
│   │
```
## Usage
Our trained models can be downloaded from [link](https://drive.google.com/drive/folders/1v2yEKWAhjRFxy8hA73R85G5f530M_7zx?usp=sharing). `models15` refers to the training results on WoodScape only, while `models17` refers to the training results on the merged WoodScape and SynWoodScape dataset. 
Qualitative evaluation of distance estimation：
```
python qualitative_distance.py
```
Quantitative evaluation of distance estimation：
```
python quantitative_distance.py
```
Train the model:
```
python main.py
```
