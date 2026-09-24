"""Neural-network inference for hip adduction and flexion moments."""

import os
import pickle
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks

try:
    from .const import (
        ACC_ALL,
        FOOT,
        GYR_ALL,
        HEIGHT_LOC,
        IMU_FIELDS,
        IMU_LIST,
        MAX_BUFFER_LEN,
        OUTPUT_DIM,
        PREDICTION_DONE,
        WAIT_PREDICTION,
        WEIGHT_LOC,
    )
except ImportError:  # Support loading the app directory as a script.
    from const import (
        ACC_ALL,
        FOOT,
        GYR_ALL,
        HEIGHT_LOC,
        IMU_FIELDS,
        IMU_LIST,
        MAX_BUFFER_LEN,
        OUTPUT_DIM,
        PREDICTION_DONE,
        WAIT_PREDICTION,
        WEIGHT_LOC,
    )


DROPOUT = 0.3
FCNN_UNIT = 32
CNN_UNIT = 72


class OutNet(nn.Module):
    """Map CNN features to a complete two-channel moment trajectory."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, FCNN_UNIT, bias=True)
        self.linear_2 = nn.Linear(FCNN_UNIT, output_dim, bias=True)
        self.relu = nn.ReLU()
        nn.init.xavier_normal_(self.linear_1.weight)
        nn.init.xavier_normal_(self.linear_2.weight)

    def forward(self, sequence, _others):
        return self.linear_2(self.relu(self.linear_1(sequence)))


class CNN_(nn.Module):
    """CNN architecture used by the supplied left- and right-side weights."""

    def __init__(self, acc_dim, gyr_dim, output_dim=OUTPUT_DIM):
        super().__init__()
        input_dim = acc_dim + gyr_dim
        self.conv_1 = nn.Conv1d(input_dim, CNN_UNIT, kernel_size=5)
        self.pooling = nn.MaxPool1d(kernel_size=2)
        self.conv_2 = nn.Conv1d(CNN_UNIT, CNN_UNIT * 2, kernel_size=3)

        conv1_length = MAX_BUFFER_LEN - 4
        pool1_length = conv1_length // 2
        conv2_length = pool1_length - 2
        pool2_length = conv2_length // 2
        fc_input_dim = CNN_UNIT * 2 * pool2_length

        self.out_net = OutNet(fc_input_dim, output_dim * MAX_BUFFER_LEN)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=DROPOUT)
        self.batchnorm_1 = nn.BatchNorm1d(CNN_UNIT)
        self.batchnorm_2 = nn.BatchNorm1d(CNN_UNIT * 2)
        self.flatten = nn.Flatten()

    def set_scalers(self, scalers):
        self.scalers = scalers

    def forward(self, acc_x, gyr_x, others, _lengths):
        output = torch.cat([acc_x, gyr_x], dim=2).permute(0, 2, 1)
        output = self.pooling(self.batchnorm_1(self.relu(self.conv_1(output))))
        output = self.pooling(self.batchnorm_2(self.relu(self.conv_2(output))))
        output = self.dropout(output)
        output = self.out_net(self.flatten(output), others)
        return output.view(-1, MAX_BUFFER_LEN, OUTPUT_DIM)


class MomentPrediction:
    """Buffer IMU samples and estimate HAM/HFM for completed gait cycles."""

    def __init__(
        self,
        weight,
        height,
        test_side="right",
        feedback_target_moment="HAM",
        find_max=True,
    ):
        if test_side not in {"right", "left"}:
            raise ValueError("test_side must be 'right' or 'left'")
        if feedback_target_moment not in {"HAM", "HFM"}:
            raise ValueError("feedback_target_moment must be 'HAM' or 'HFM'")

        self.data_buffer = deque(maxlen=MAX_BUFFER_LEN)
        self.data_margin_before_step = 20
        self.data_margin_after_step = 20
        self.data_array_fields = [
            f"{axis}_{sensor}" for sensor in IMU_LIST for axis in IMU_FIELDS
        ]
        self.feedback_target_moment = feedback_target_moment
        self.find_max = find_max

        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, "models", f"model_state_{test_side}.pth")
        scaler_path = os.path.join(base_path, "models", f"model_scalers_{test_side}.pkl")

        self.model = CNN_(acc_dim=12, gyr_dim=12)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        with open(scaler_path, "rb") as scaler_file:
            self.model.set_scalers(pickle.load(scaler_file))

        self.model.acc_col_loc = [self.data_array_fields.index(field) for field in ACC_ALL]
        self.model.gyr_col_loc = [self.data_array_fields.index(field) for field in GYR_ALL]

        anthropometrics = np.zeros(
            (1, MAX_BUFFER_LEN, OUTPUT_DIM), dtype=np.float32
        )
        anthropometrics[:, :, WEIGHT_LOC] = weight
        anthropometrics[:, :, HEIGHT_LOC] = height
        self.model_inputs = {
            "others": torch.from_numpy(anthropometrics),
            "step_length": None,
            "input_acc": None,
            "input_gyr": None,
        }

    def _find_step_peak(self, prediction, step_length, moment_column, find_max):
        """Return the target extreme and its index within one predicted step."""
        if step_length < 3:
            return -1, 0.0

        if moment_column == 0 and find_max:
            start = int(step_length * 0.20)
            end = int(step_length * 0.70)
            if end <= start + 1:
                return -1, 0.0

            peaks, _ = find_peaks(
                prediction[start:end, moment_column],
                height=1,
                distance=10,
                prominence=0.1,
            )
            if peaks.size == 0:
                return -1, 0.0
            index = start + int(peaks[0])
            return index, float(prediction[index, moment_column])

        values = prediction[:step_length, moment_column]
        index = int(np.argmax(values) if find_max else np.argmin(values))
        return index, float(values[index])

    def update_stream(self, data, gait_phase):
        """Add one sample and return zero or one time-aligned output samples."""
        self.data_buffer.append([data, 0.0, 0.0, 0, -1, 0.0])
        package = data[FOOT]["Package"]

        if gait_phase.current_phase == WAIT_PREDICTION:
            samples_after_off = package - gait_phase.off_package
            if samples_after_off >= self.data_margin_after_step - 1:
                step_length = int(
                    gait_phase.off_package
                    - gait_phase.strike_package
                    + self.data_margin_before_step
                    + self.data_margin_after_step
                )
                if 0 < step_length <= len(self.data_buffer):
                    inputs = self.transform_input(step_length)
                    with torch.inference_mode():
                        prediction = self.model(
                            inputs["input_acc"],
                            inputs["input_gyr"],
                            inputs["others"],
                            inputs["step_length"],
                        )[0].numpy().astype(float)

                    for sample_index in range(step_length):
                        self.data_buffer[-step_length + sample_index][1:3] = prediction[
                            sample_index, :
                        ]
                    for sample_index in range(
                        self.data_margin_before_step,
                        step_length - self.data_margin_after_step,
                    ):
                        self.data_buffer[-step_length + sample_index][3] = 1

                    moment_column = 0 if self.feedback_target_moment == "HAM" else 1
                    peak_index, peak_value = self._find_step_peak(
                        prediction,
                        step_length,
                        moment_column=moment_column,
                        find_max=self.find_max,
                    )
                    self.data_buffer[0][4] = peak_index
                    self.data_buffer[0][5] = peak_value

                gait_phase.current_phase = PREDICTION_DONE

        if len(self.data_buffer) == MAX_BUFFER_LEN:
            return [self.data_buffer.popleft()]
        return []

    def transform_input(self, step_length):
        """Convert buffered sensor dictionaries to normalized model tensors."""
        rows = []
        for sample in list(self.data_buffer)[-step_length:]:
            row = []
            for sensor_index in range(len(IMU_LIST)):
                row.extend(sample[0][sensor_index][field] for field in IMU_FIELDS)
            rows.append(row)

        data = np.zeros(
            (MAX_BUFFER_LEN, len(self.data_array_fields)), dtype=np.float32
        )
        data[:step_length, :] = np.asarray(rows, dtype=np.float32)

        acc_locations = self.model.acc_col_loc
        gyr_locations = self.model.gyr_col_loc
        data[:, acc_locations] = self._normalize(
            data[:, acc_locations], self.model.scalers["input_acc"]
        )
        data[:, gyr_locations] = self._normalize(
            data[:, gyr_locations], self.model.scalers["input_gyr"]
        )

        self.model_inputs["input_acc"] = torch.from_numpy(
            np.expand_dims(data[:, acc_locations], axis=0)
        )
        self.model_inputs["input_gyr"] = torch.from_numpy(
            np.expand_dims(data[:, gyr_locations], axis=0)
        )
        self.model_inputs["step_length"] = torch.tensor(
            [step_length], dtype=torch.int32
        )
        return self.model_inputs

    @staticmethod
    def _normalize(data, scaler):
        """Apply a fitted scaler while preserving zero-padded rows."""
        normalized = data.copy()
        zero_rows = (normalized == 0.0).all(axis=1)
        normalized[zero_rows, :] = np.nan
        normalized = scaler.transform(normalized)
        normalized[np.isnan(normalized)] = 0.0
        return normalized
