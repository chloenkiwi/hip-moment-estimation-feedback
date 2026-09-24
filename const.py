"""Shared constants for sensor processing and model inference."""

PRE_STATIONARY, POST_STATIONARY, WAIT_PREDICTION, PREDICTION_DONE = range(300, 304)

IMU_LIST = ["FOOT", "THIGH", "PELVIS", "CHEST"]
FOOT, THIGH, PELVIS, CHEST = range(len(IMU_LIST))
IMU_FIELDS = ["AccelX", "AccelY", "AccelZ", "GyroX", "GyroY", "GyroZ"]

ACC_ALL = [
    f"{field}_{sensor}" for sensor in IMU_LIST for field in IMU_FIELDS[:3]
]
GYR_ALL = [
    f"{field}_{sensor}" for sensor in IMU_LIST for field in IMU_FIELDS[3:]
]

MAX_BUFFER_LEN = 176
OUTPUT_DIM = 2
GRAVITY = 9.81

SUBJECT_WEIGHT, SUBJECT_HEIGHT = STATIC_DATA = ["weight", "height"]
WEIGHT_LOC = STATIC_DATA.index(SUBJECT_WEIGHT)
HEIGHT_LOC = STATIC_DATA.index(SUBJECT_HEIGHT)
