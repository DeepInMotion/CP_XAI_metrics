cp-xai-metrics/
  ├── README.md
  ├── metrics.py    # Code for XAI metrics
  └── window_selections		# Place extracted windows from the infant pose data here

# Cerebral Palsy XAI Metrics

A Python module for evaluating XAI techniques in our CP prediction ensemble.

## Overview

This repository contains metrics for quantifying the faithfulness and stability of explanations in AI models that predict cerebral palsy risk from infant pose data. These metrics help evaluate how well explainers like CAM or Grad-CAM accurately reflect the model's decision-making process.

## Metrics Included

### Faithfulness Metrics

- **PGI (Prediction Gap on Important feature perturbation)**: Measures how perturbations of important features impact model output. Higher values indicate better faithfulness.
- **PGU (Prediction Gap on Unmportant feature perturbation)**: Measures how perturbations of unimportant features impact model output. Lower values indicate better faithfulness.

### Stability Metrics

- **ROS (Robustness to Output Stability)**: Measures explanation stability relative to output changes.
- **RIS (Robustness to Input Stability)**: Measures explanation stability relative to input changes. Calculated for three input types:
  - RISp: Position input
  - RISv: Velocity input
  - RISb: Bone length input
- **RRS (Robustness to Representation Stability)**: Measures explanation stability relative to internal representation changes.
- **Stability values closer to zero indicate robust explanations.

## Usage

This module is designed to be used with our CP prediction ensemble. Create a wrapper function that interfaces with the model and passes the results to our metrics evaluation.

```

### Required Function Interface

The prediction function accepts these parameters:
- `tracking_coords`: Numpy array of body joint coordinates
- `body_parts`: List of body part names
- `frame_rate`: Frame rate of the original video
- `total_frames`: Total number of frames in the video
- `pred_frame_rate`: Frame rate for prediction (default: 30.0)
- `pred_interval_seconds`: Time between prediction windows (default: 2.5)
- `window_stride`: Stride for median filter (default: 2)
- `num_models`: Number of models in ensemble (default: 10)
- `num_portions`: Number of data portions (default: 7)
- `prediction_threshold`: Threshold for positive prediction (default: 0.350307)
- `xai_technique`: XAI technique to evaluate ('cam' or 'gradcam')

And return:
- `window_cams`: activation maps for each window (CAM or Grad-CAM)
- `window_cp_risks`: CP risk predictions for each window
- `inputs`: Model inputs
- `internal_reps`: Internal model representations

## Required Data Structure (Our data)

- `video_fps_dict.json`: Dictionary mapping video filenames to their frame rates
- `window_selections/`: Directory containing CSV files with tracking coordinates

## Citation

If you use our code in your research, please cite:

```
@article{pellano2025evaluating,
  title={Evaluating explainable ai methods in deep learning models for early detection of cerebral palsy},
  author={Pellano, Kimji N and Str{\"u}mke, Inga and Groos, Daniel and Adde, Lars and Ihlen, Espen Alexander F},
  journal={IEEE Access},
  year={2025},
  publisher={IEEE}
}
```