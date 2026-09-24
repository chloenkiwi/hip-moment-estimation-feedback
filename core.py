"""SAGE application entry point for real-time hip-moment estimation."""

from sage.base_app import BaseApp

try:
    from .gait_phase import GaitPhase, Transitions
    from .hip_moments import MomentPrediction
except ImportError:  # Support loading the app directory as a script.
    from gait_phase import GaitPhase, Transitions
    from hip_moments import MomentPrediction


class Core(BaseApp):
    """Run gait detection, moment inference, feedback, and data streaming."""

    def __init__(self, my_sage):
        super().__init__(my_sage, __file__)

        self.test_side = self.config["test_side"]
        self.feedback_target_moment = self.config["feedback_target_moment"]
        self.feedback_at_peak = self.config["feedback_phase_target"] == "end_stance"
        self.min_threshold = self.config["min_threshold"]
        self.max_threshold = self.config["max_threshold"]

        self.gait_phase = GaitPhase()
        self.moment_prediction = MomentPrediction(
            weight=self.config["weight"],
            height=self.config["height"],
            test_side=self.test_side,
            feedback_target_moment=self.feedback_target_moment,
            find_max=self.config["max_min_window"] == "max",
        )

        self.foot_nodenum = self.info["sensors"].index("FOOT")
        self.thigh_nodenum = self.info["sensors"].index("THIGH")
        self.pelvis_nodenum = self.info["sensors"].index("PELVIS")

        if self.info["feedback"]:
            self.feedback_min_node = self.info["feedback"].index("feedback_min")
            self.feedback_max_node = self.info["feedback"].index("feedback_max")
            self.feedback_on = self.my_sage.feedback_vibration_on
            self.feedback_off = self.my_sage.feedback_vibration_off
        else:
            self.feedback_min_node = self.thigh_nodenum
            self.feedback_max_node = self.pelvis_nodenum
            self.feedback_on = self.my_sage.sensor_vibration_on
            self.feedback_off = self.my_sage.sensor_vibration_off

        self.time_now = 0.0
        self.feedback_min = 0
        self.feedback_max = 0

    def reset_feedback(self):
        """Turn off both feedback channels at the start of a gait cycle."""
        self.feedback_min = 0
        self.feedback_max = 0
        self.feedback_off(self.feedback_min_node)
        self.feedback_off(self.feedback_max_node)

    def give_feedback(self, value):
        """Apply mutually exclusive lower- or upper-threshold feedback."""
        if value < self.min_threshold:
            self.feedback_on(self.feedback_min_node, self.info["pulse_length"])
            self.feedback_off(self.feedback_max_node)
            self.feedback_min, self.feedback_max = 1, 0
        elif value > self.max_threshold:
            self.feedback_on(self.feedback_max_node, self.info["pulse_length"])
            self.feedback_off(self.feedback_min_node)
            self.feedback_min, self.feedback_max = 0, 1
        else:
            self.feedback_off(self.feedback_min_node)
            self.feedback_off(self.feedback_max_node)
            self.feedback_min, self.feedback_max = 0, 0

    def run_in_loop(self):
        """Process one incoming sample from the SAGE runtime."""
        sensor_data = self.my_sage.get_next_data()
        self.gait_phase.update_gaitphase(sensor_data[self.foot_nodenum])

        if (
            self.config["feedback_enabled"]
            and self.gait_phase.transition == Transitions.SWING_TO_STANCE
        ):
            self.reset_feedback()

        predictions = self.moment_prediction.update_stream(sensor_data, self.gait_phase)
        for raw_data, ham, hfm, stance_flag, peak_idx, peak_value in predictions:
            self.time_now += 0.01

            if self.config["feedback_enabled"]:
                if peak_idx >= 0:
                    print(
                        f"[PEAK] t={self.time_now:.2f}s, "
                        f"index={peak_idx}, value={peak_value:.3f}"
                    )
                    self.give_feedback(peak_value)
                elif not self.feedback_at_peak:
                    current_value = ham if self.feedback_target_moment == "HAM" else hfm
                    self.give_feedback(current_value)

            output = {
                "time": [self.time_now],
                "HAM": [ham],
                "HFM": [hfm],
                "Stance_Flag": [stance_flag],
                "gait_transition": [self.gait_phase.transition.value],
                "gait_transition_name": [self.gait_phase.transition.name],
                "feedback_state_min": [self.feedback_min],
                "feedback_state_max": [self.feedback_max],
                "min_threshold": [float(self.min_threshold)],
                "max_threshold": [float(self.max_threshold)],
            }
            self.my_sage.send_stream_data(raw_data, output)
            self.my_sage.save_data(raw_data, output)

        return True
