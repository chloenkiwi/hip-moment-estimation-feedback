# 074_HFMHAM
# Hip Moment Estimation and Feedback

Real-time estimation of hip adduction moment (HAM) and hip flexion moment (HFM) from four wearable IMUs, with optional vibrotactile feedback through the SAGE runtime.

## Features

- Processes four IMUs at 100 Hz
- Detects gait phases from the foot IMU
- Estimates HAM and HFM using a pretrained convolutional neural network
- Provides separate pretrained weights for the left and right sides
- Supports immediate feedback or stance-phase peak feedback
- Streams and saves predictions through the SAGE application interface

## Sensor setup

The sensor order in `info.json` must match the order supplied by the SAGE runtime:

| Index | Sensor | Placement |
|---:|---|---|
| 0 | `FOOT` | Foot on the selected test side |
| 1 | `THIGH` | Thigh on the selected test side |
| 2 | `PELVIS` | Posterior pelvis |
| 3 | `CHEST` | Chest/trunk |

Each sensor sample must provide:

```text
Package, AccelX, AccelY, AccelZ, GyroX, GyroY, GyroZ
```

`Package` is used to calculate sample intervals between gait events. Sensor orientation must be consistent with the orientation used to train and validate the models.

## Requirements

- A SAGE runtime that provides `sage.base_app.BaseApp`
- Python 3.10 or later
- PyTorch
- NumPy
- SciPy
- scikit-learn

Install the public Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

SAGE is a hardware-specific runtime and is not installed by `requirements.txt`. Follow the instructions supplied with your SAGE system to install it and load this app directory. This project is not a standalone command-line program.

## Configuration

Runtime settings are stored in `config.json`:

| Setting | Values | Description |
|---|---|---|
| `feedback_enabled` | `true`, `false` | Enable or disable vibration feedback |
| `feedback_target_moment` | `HAM`, `HFM` | Moment used for feedback |
| `test_side` | `right`, `left` | Select the matching model and sensor side |
| `feedback_phase_target` | `none`, `end_stance` | Immediate or stance-phase peak feedback |
| `max_min_window` | `max`, `min` | Select a maximum or minimum for peak feedback |
| `min_threshold` | number | Trigger `feedback_min` below this value |
| `max_threshold` | number | Trigger `feedback_max` above this value |
| `weight` | kg | Participant mass |
| `height` | m | Participant height |
| `save_mode` | `csv`, `h5`, `xlsx` | Output format handled by SAGE |
| `trial_name` | string | Trial identifier |

Device addresses are intentionally empty in the published configuration. Pair the sensors and feedback units in SAGE before use.

## Model inference

The input contains 12 acceleration and 12 angular-velocity channels. A detected gait cycle is placed in a fixed 176-sample window and normalized with the supplied scikit-learn scalers. The CNN returns a `176 × 2` trajectory in the order `[HAM, HFM]`.

The selected side determines the files loaded at startup:

| Side | Model weights | Scaler |
|---|---|---|
| Right | `models/model_state_right.pth` | `models/model_scalers_right.pkl` |
| Left | `models/model_state_left.pth` | `models/model_scalers_left.pkl` |

The model runs on the CPU. Changes to sensor order, input fields, network architecture, or the 176-sample window require compatible model files.

## Outputs

| Field | Description |
|---|---|
| `time` | Application time in seconds |
| `HAM` | Predicted hip adduction moment |
| `HFM` | Predicted hip flexion moment |
| `Stance_Flag` | Valid stance-region marker |
| `gait_transition` | Numeric gait-transition state |
| `gait_transition_name` | Human-readable transition name |
| `feedback_state_min` | Lower-threshold feedback state |
| `feedback_state_max` | Upper-threshold feedback state |
| `min_threshold` | Active lower threshold |
| `max_threshold` | Active upper threshold |
