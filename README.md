# weakly-supervised-fisheye-distance
This is the official repository of paper "Weakly Supervised Monocular Fisheye Camera Distance Estimation with Segmentation Constraints". Electronics, 2025. This code is based on [WoodScape](https://github.com/valeoai/WoodScape/tree/master), with modifications.
## Methodology
![pipeline](https://github.com/user-attachments/assets/8ed8ba2c-97f2-454b-9cbd-a062589f9110)
## Datasets
[WoodScape](https://woodscape.valeo.com/woodscape/)
[SynWoodScape](https://drive.google.com/drive/folders/1N5rrySiw1uh9kLeBuOblMbXJ09YsqO7I)
## Data organization
```
WoodScape_ICCV19
└───rgb_images
│   │   00001_[CAM].png
│   │   00002_[CAM].png
|   |   ...
│   │
└───previous_images
│   │   00001_[CAM]_prev.png
│   │   00002_[CAM]_prev.png
|   |   ...
│   │
└───segment_annotations
        │   instance
        │   │   00001_[CAM].png
        │   │   00002_[CAM].png
        |   |   ...
        │   semantic
        │   │   00001_[CAM].png
        │   │   00002_[CAM].png
        |   |   ...
│   │
└───vehicle_data
        │   previous_images
        │   │   00001_[CAM].png
        │   │   00002_[CAM].png
        |   |   ...
        │   rgb_images
        │   │   00001_[CAM].png
        │   │   00002_[CAM].png
        |   |   ...
│   │
└───flow_annotations_flowformersintel
        │   gtLabels
        │   │   00001_[CAM].flo
        │   │   00002_[CAM].flo
        |   |   ...
        │   rgbLabels
        │   │   00001_[CAM].png
        │   │   00002_[CAM].png
        |   |   ...
│   │
└───calibration_data
│   │   00001_[CAM].json
│   │   00002_[CAM].json
|   |   ...
│   │
└───masks
│   │
└───raw_data
│   │   08234_FV.npy
|   |   ...
│   │
```
