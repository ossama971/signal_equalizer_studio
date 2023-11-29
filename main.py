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
from pydub import AudioSegment

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
        self.output : Signal = None
        self.output_playing = False
        self.output_current_timer = QTimer(self)
        self.output_current_timer.timeout.connect(self.update_output_current_timer)
        self.output_current_time = 0

        self.frequencies = None
        self.fourier_transform = None
        self.last_index = 0
        self.current_timer = QTimer(self)
        self.current_timer.timeout.connect(self.update_current_timer)
        self.current_time = 0

    def _initialize_signals_slots(self):
        self.import_action.triggered.connect(self._import_signal_file)
        self.input_play_button.pressed.connect(self.play_time_signal)
        self.output_play_button.pressed.connect(self.play_time_output)
        self.input_slider.valueChanged.connect(self._on_input_slider_change)
        self.output_slider.valueChanged.connect(self._on_output_slider_change)
        self.control_slider_1.valueChanged.connect(self.generate_output_signal)
        


    def _on_input_slider_change(self, value):
        if self.signal:
            self.current_time = value / 1000
            self.update_current_timer()

    def _on_output_slider_change(self, value):
        if self.output:
            self.output_current_time = value / 1000
            self.update_output_current_timer()

    def _import_signal_file(self):
        self.signal: Signal = get_signal_from_file(self)

        # plot time graph
        pen_c = pg.mkPen(color=(255, 255, 255))
        self.input_signal_graph.plot(self.signal.x_vec, self.signal.y_vec, pen=pen_c)

        # plot initial output time graph
        # self.output_signal_graph.plot(self.signal.x_vec, self.signal.y_vec, pen=pen_c) 

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
        self.frequencies, self.fourier_transform = self.apply_fourier_transform()
        pen_c = pg.mkPen(color=(255, 255, 255))
        self.frequency_graph.plot(self.frequencies, abs(self.fourier_transform), pen=pen_c)

    def generatePgColormap(self, cm_name):
        colormap = plt.get_cmap(cm_name)
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        return lut


    def plot_input_spectrograph(self):


        # Compute the spectrogram using scipy's spectrogram function
        amplitude = self.signal.y_vec
        sampling_rate = self.signal.get_sampling_frequency()
        f, t, Sxx = spectrogram(amplitude, fs=sampling_rate)

        # Plot the spectrogram
        np.seterr(divide='ignore')
        plt.pcolormesh(10 * np.log10(Sxx), shading='auto')
        plt.title('Spectrograph')
        plt.xlabel('time')
        plt.ylabel('frequency')
        plt.colorbar(label='Intensity (dB)')  # Add colorbar

        self.input_spectrogram_graph.clear()

        # Create an ImageItem to display the spectrogram
        spectrogram_image = pg.ImageItem()

        lut = self.generatePgColormap('viridis')
        spectrogram_image.setLookupTable(lut)

        # Set the spectrogram data and scaling
        spectrogram_image.setImage(10 * np.log10(Sxx.T), autoLevels=True, autoDownsample=True)
        self.input_spectrogram_graph.addItem(spectrogram_image)

        # Set labels for the axes
        self.input_spectrogram_graph.setLabel('left', 'Frequency', units='Hz')
        self.input_spectrogram_graph.setLabel('bottom', 'Time', units='s')


    def apply_fourier_transform(self):
        if self.signal.audio:
            sampling_frequency = self.signal.audio.frame_rate
        else:
            sampling_frequency = 1000
        # Frequency domain representation
        amplitude = self.signal.y_vec
        fourier_transform = np.fft.fft(amplitude) / len(amplitude)  # Normalize amplitude
        
        # fourier_transform = fourier_transform[range(int(len(amplitude) / 2))]  # Exclude sampling frequency
        
        tp_count = len(amplitude)

        values = np.arange(int(tp_count))

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



    def generate_output_signal(self):
        self.output_signal_graph.clear()
        pen_c = pg.mkPen(color=(255, 255, 255))
        
        # Generate output using inverse Fourier transform of self.frequency and self.fourier_transform
        if self.fourier_transform is not None:
            y_vec = np.fft.ifft(self.fourier_transform).real  # take the real part of the complex numbers
         


            x_vec = self.signal.x_vec
            self.output_signal_graph.plot(x_vec, y_vec, pen=pen_c)
            self.output_signal_graph.repaint()
            
            # Normalization
            max_amplitude = np.max(np.abs(y_vec))
            if max_amplitude > 0:
                y_vec /= max_amplitude

            audio = AudioSegment(
                y_vec.astype(np.int16).tobytes(),
                frame_rate=self.signal.audio.frame_rate,
                sample_width=2,
                channels=1  
            )
            self.output = Signal(x_vec, y_vec, audio)
            self.output_slider.setMinimum(0)
            self.output_slider.setMaximum(int(self.output.x_vec[-1] * 1000))
            self.output_slider.setValue(0)
            self.output_total_time.setText(
            f'{str(math.floor(self.output.x_vec[-1] / 60)).zfill(2)}:{str(math.floor(self.output.x_vec[-1]) % 60).zfill(2)}')
            self.plot_output_spectograpgh()



    
    def play_ouput(self):
        if self.output.audio:
            self.output_playing = True
            output_final_index = np.abs(self.output.x_vec - self.output_current_time).argmin()
            sd.play(self.output.y_vec[output_final_index:], self.output.audio.frame_rate * 2)
            sd.wait()
            self.output_current_timer.stop()
            self.output_playing = False
            if output_final_index >= len(self.output.x_vec) - 100:
                print('rewind')
                self.output_play_button.setText('Rewind')
                self.output_current_time = 0;
    
    def play_time_output(self):
        if self.output is not None and self.signal.audio:
            if self.output_playing:
                sd.stop()
                self.output_playing = False
                self.output_play_button.setText('Play')
                self.output_current_timer.stop()
            else:
                self.output_audio_thread = Thread(target=self.play_ouput)
                self.output_audio_thread.start()
                self.output_current_timer.start(10)
                self.output_playing = True
                self.output_play_button.setText('Pause')

    def update_output_current_timer(self):
        self.output_current_time += 0.01;
        self.current_output_time.setText(
            f'{str(math.floor(self.output_current_time / 60)).zfill(2)}:{str(math.floor(self.output_current_time) % 60).zfill(2)}')
        self.output_slider.blockSignals(True)
        self.output_slider.setValue(math.ceil(self.output_current_time * 1000))
        self.output_slider.blockSignals(False)
        self.output_slider.repaint()


    def plot_output_spectograpgh(self):
        # Compute the spectrogram using scipy's spectrogram function
        amplitude = self.output.y_vec
        sampling_rate = self.output.get_sampling_frequency()
        _, _, Sxx = spectrogram(amplitude, fs=sampling_rate)

        # Plot the spectrogram
        np.seterr(divide='ignore')
        plt.pcolormesh(10 * np.log10(Sxx), shading='auto')
        plt.title('Spectrograph')
        plt.xlabel('time')
        plt.ylabel('frequency')
        plt.colorbar(label='Intensity (dB)')

        self.output_spectrogram_graph.clear()

        spectrogram_image = pg.ImageItem()
        lut = self.generatePgColormap('viridis')
        spectrogram_image.setLookupTable(lut)

        # Set the spectrogram data and scaling
        spectrogram_image.setImage(10 * np.log10(Sxx.T), autoLevels=True, autoDownsample=True)
        self.output_spectrogram_graph.addItem(spectrogram_image)

        # Set labels for the axes
        self.output_spectrogram_graph.setLabel('left', 'Frequency', units='Hz')
        self.output_spectrogram_graph.setLabel('bottom', 'Time', units='s')
    
    
    

        


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
