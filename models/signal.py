from enum import Enum
import numpy as np

class SignalType(Enum):
    CONTINUOUS = 0
    DISCRETE = 1


class Signal:
    def __init__(self, x_vec, y_vec, audio=None, signal_type: SignalType = SignalType.CONTINUOUS) -> None:
        self.x_vec = x_vec
        self.y_vec = y_vec
        self.audio = audio
        self.signal_type = signal_type

    def get_sampling_frequency(self):
        if self.audio:
            sampling_frequency = self.audio.frame_rate
        else:
            time_difference = np.diff(self.x_vec)
            average_time_difference = np.mean(time_difference)
            sampling_frequency = 1 / average_time_difference

        return sampling_frequency


