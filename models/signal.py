from enum import Enum
import numpy as np

class SignalType(Enum):
    CONTINUOUS = 0
    DISCRETE = 1


class Signal:
    def __init__(self, x_vec, y_vec,audio=None, signal_type: SignalType = SignalType.CONTINUOUS, is_playing = False, current_index = 0, current_time = 0) -> None:
        self.x_vec = x_vec
        self.y_vec = y_vec
        self.audio = audio
        self.signal_type = signal_type
        self.is_playing = is_playing
        self.current_index = current_index
        self.current_time = current_time

    def get_sampling_frequency(self):
        if self.audio:
            sampling_frequency = self.audio.frame_rate
        else:
            time_difference = np.diff(self.x_vec)
            average_time_difference = np.mean(time_difference)
            sampling_frequency = 1 / average_time_difference

        return sampling_frequency


