import os
import sys
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
import pyqtgraph as pg
import sounddevice as sd
from threading import Thread
import numpy as np
import math
from PyQt6.QtCore import QTimer
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

from helpers.get_signal_from_file import get_signal_from_file
from models.signal import Signal

mainwindow_ui_file_path = os.path.join(os.path.dirname(__file__), 'views', 'mainwindow.ui')
uiclass, baseclass = pg.Qt.loadUiType(mainwindow_ui_file_path)


class MainWindow(uiclass, baseclass):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Signal Equalizer Studio")
        self._initialize_signals_slots()
        self.signal = None
        self.playing = False
        self.last_index = 0
        self.current_timer = QTimer()
        self.current_timer.timeout.connect(self.update_current_timer)
        self.current_time = 0

    def _initialize_signals_slots(self):
        self.import_action.triggered.connect(self._import_signal_file)
        self.input_play_button.pressed.connect(self.play_time_signal)
        self.input_slider.valueChanged.connect(self._on_input_slider_change)

    def _on_input_slider_change(self, value):
        if self.signal:
            self.current_time = value / 1000
            self.update_current_timer()

    def _import_signal_file(self):
        self.signal: Signal = get_signal_from_file(self)

        # plot time graph
        pen_c = pg.mkPen(color=(255, 255, 255))
        self.input_signal_graph.plot(self.signal.x_vec, self.signal.y_vec, pen=pen_c)

        # plot initial output time graph
        self.output_signal_graph.plot(self.signal.x_vec, self.signal.y_vec, pen=pen_c)

        self.input_slider.setMinimum(0)
        self.input_slider.setMaximum(int(self.signal.x_vec[-1] * 1000))
        self.input_slider.setValue(0)
        self.input_total_time.setText(
            f'{str(math.floor(self.signal.x_vec[-1] / 60)).zfill(2)}:{str(math.floor(self.signal.x_vec[-1]) % 60).zfill(2)}')

        # plot input frequency graph
        self.plot_input_frequency()

        # plot input spectrograph
        self.plot_input_spectrograph()

    def plot_input_frequency(self):
        frequencies, fourier_transform = self.apply_fourier_transform()
        pen_c = pg.mkPen(color=(255, 255, 255))
        self.frequency_graph.plot(frequencies, abs(fourier_transform), pen=pen_c)

    def plot_input_spectrograph(self):
        """
        Plot the spectrogram of a signal.

        Parameters:
        - time (numpy array): Array representing the time values.
        - amplitude (numpy array): Array representing the amplitude values.
        - sampling_rate (float, optional): The sampling rate of the signal. Default is 1.0.
        - title (str, optional): Title of the plot. Default is 'Spectrogram'.
        - xlabel (str, optional): Label for the x-axis. Default is 'Time'.
        - ylabel (str, optional): Label for the y-axis. Default is 'Frequency'.
        """
        # Compute the spectrogram using scipy's spectrogram function
        amplitude = self.signal.y_vec
        sampling_rate = self.signal.get_sampling_frequency()
        f, t, Sxx = spectrogram(amplitude, fs=sampling_rate)

        # Plot the spectrogram
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='auto')
        # plt.title(title)
        # plt.xlabel(xlabel)
        # plt.ylabel(ylabel)
        plt.colorbar(label='Intensity (dB)')  # Add colorbar

        # Show the plot
        plt.show()


    def apply_fourier_transform(self):
        if self.signal.audio:
            sampling_frequency = self.signal.audio.frame_rate
        else:
            sampling_frequency = 1000
        # Frequency domain representation
        amplitude = self.signal.y_vec
        fourier_transform = np.fft.fft(amplitude) / len(amplitude)  # Normalize amplitude

        fourier_transform = fourier_transform[range(int(len(amplitude) / 2))]  # Exclude sampling frequency
        tp_count = len(amplitude)

        values = np.arange(int(tp_count / 2))

        time_period = tp_count / sampling_frequency

        frequencies = values / time_period

        return frequencies, fourier_transform

    def play_audio(self):
        if self.signal.audio:
            self.playing = True
            final_index = np.abs(self.signal.x_vec - self.current_time).argmin()
            sd.play(self.signal.y_vec[final_index:], self.signal.audio.frame_rate * 2)
            sd.wait()
            self.current_timer.stop()
            self.playing = False
            final_index = np.abs(self.signal.x_vec - self.current_time).argmin()
            if final_index >= len(self.signal.x_vec) - 100:
                self.input_play_button.setText('Rewind')
                self.current_time = 0;

    def play_time_signal(self):
        if self.signal.audio:
            if self.playing:
                sd.stop()
                self.playing = False
                self.input_play_button.setText('Play')
                self.current_timer.stop()

            else:
                self.audio_thread = Thread(target=self.play_audio)
                self.audio_thread.start()
                self.current_timer.start(10)
                self.playing = True
                self.input_play_button.setText('Pause')

    def update_current_timer(self):
        self.current_time += 0.01;
        self.current_input_time.setText(
            f'{str(math.floor(self.current_time / 60)).zfill(2)}:{str(math.floor(self.current_time) % 60).zfill(2)}')
        self.input_slider.blockSignals(True)
        self.input_slider.setValue(math.ceil(self.current_time * 1000))
        self.input_slider.blockSignals(False)
        self.input_slider.repaint()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
