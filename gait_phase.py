"""Rule-based gait-phase detection from the foot IMU."""

import itertools
from collections import deque
from enum import Enum

import numpy as np

try:
    from .const import (
        GRAVITY,
        MAX_BUFFER_LEN,
        POST_STATIONARY,
        PREDICTION_DONE,
        PRE_STATIONARY,
        WAIT_PREDICTION,
    )
except ImportError:  # Support loading the app directory as a script.
    from const import (
        GRAVITY,
        MAX_BUFFER_LEN,
        POST_STATIONARY,
        PREDICTION_DONE,
        PRE_STATIONARY,
        WAIT_PREDICTION,
    )


class Transitions(Enum):
    NO_CHANGE = 0
    SWING_TO_STANCE = 1
    STANCE_TO_PREDICTION = 2
    PREDICTION_TO_SWING = 3


class GaitPhase:
    """Track gait events using acceleration and angular-velocity thresholds."""

    def __init__(self):
        self.acc_threshold = 1.2
        self.gyro_threshold = np.rad2deg(2.6)
        self.current_phase = PRE_STATIONARY
        self.last_phase = PRE_STATIONARY
        self.transition = Transitions.NO_CHANGE

        self.stationary_window = 10
        self.acc_magnitude = deque(maxlen=self.stationary_window)
        self.gyro_magnitude = deque(maxlen=self.stationary_window)
        self.data_buffer = deque(maxlen=250)
        self.strike_package = 0
        self.off_package = 0

    def update_gaitphase(self, foot_data):
        """Update the gait state with one foot-IMU sample."""
        self._update_transition()
        self.data_buffer.append((foot_data["Package"], foot_data["GyroX"]))

        if self.current_phase == PRE_STATIONARY:
            self._update_pre_stationary(foot_data)
        elif self.current_phase == POST_STATIONARY:
            self._update_post_stationary(foot_data)
        elif self.current_phase == WAIT_PREDICTION:
            self.last_phase = WAIT_PREDICTION
        elif self.current_phase == PREDICTION_DONE:
            self.current_phase = PRE_STATIONARY
            self.transition = Transitions.PREDICTION_TO_SWING

    def _update_transition(self):
        if self.current_phase == self.last_phase:
            self.transition = Transitions.NO_CHANGE
        elif self.last_phase == PRE_STATIONARY and self.current_phase == POST_STATIONARY:
            self.transition = Transitions.SWING_TO_STANCE
        elif self.last_phase == POST_STATIONARY and self.current_phase == WAIT_PREDICTION:
            self.transition = Transitions.STANCE_TO_PREDICTION

    def _update_pre_stationary(self, foot_data):
        self.last_phase = PRE_STATIONARY
        acceleration = [
            foot_data["AccelX"],
            foot_data["AccelY"],
            foot_data["AccelZ"],
        ]
        angular_velocity = [
            foot_data["GyroX"],
            foot_data["GyroY"],
            foot_data["GyroZ"],
        ]
        self.acc_magnitude.append(abs(np.linalg.norm(acceleration) - GRAVITY))
        self.gyro_magnitude.append(np.linalg.norm(angular_velocity))

        if len(self.acc_magnitude) < self.stationary_window:
            return
        is_stationary = (
            np.asarray(self.acc_magnitude) < self.acc_threshold
        ).all() and (
            np.asarray(self.gyro_magnitude) < self.gyro_threshold
        ).all()
        if not is_stationary:
            return

        self.acc_magnitude.clear()
        self.gyro_magnitude.clear()
        for sample_index in range(len(self.data_buffer) - 1, 0, -1):
            if self.data_buffer[sample_index][1] < -self.gyro_threshold:
                search_slice = list(
                    itertools.islice(self.data_buffer, sample_index, len(self.data_buffer))
                )
                buffered_data = np.asarray(search_slice)
                strike_index = int(np.argmax(buffered_data[:, 1]))
                self.strike_package = buffered_data[strike_index, 0]
                self.data_buffer.clear()
                break
        self.current_phase = POST_STATIONARY

    def _update_post_stationary(self, foot_data):
        self.last_phase = POST_STATIONARY
        if foot_data["GyroX"] >= -self.gyro_threshold:
            return

        buffered_data = np.asarray(self.data_buffer)
        off_index = int(np.argmax(buffered_data[:, 1]))
        self.off_package = buffered_data[off_index, 0]
        if self.off_package - self.strike_package < MAX_BUFFER_LEN:
            self.current_phase = WAIT_PREDICTION
        else:
            self.current_phase = PRE_STATIONARY
