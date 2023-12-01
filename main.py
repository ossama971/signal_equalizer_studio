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
from scipy.signal import spectrogram, square, hamming,gaussian
import matplotlib.pyplot as plt
from pydub import AudioSegment

from helpers.get_signal_from_file import get_signal_from_file
from models.signal import Signal
from enum import Enum
from functools import partial

mainwindow_ui_file_path = os.path.join(os.path.dirname(__file__), 'views', 'mainwindow.ui')
uiclass, baseclass = pg.Qt.loadUiType(mainwindow_ui_file_path)


class WindowType(Enum):
    RECTANGLE = 'rectangle'
    HAMMING = 'hamming'
    HANNING = 'hanning'
    GAUSSIAN = 'gaussian'

class ModeType(Enum):
    ANIMALS = 'animals'
    MUSIC = 'music'
    UNIFORM = 'uniform'
    ECG = 'ecg'

class MainWindow(uiclass, baseclass):
    
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Signal Equalizer Studio")
        self.signal = None
        self.playing = False
        self.output : Signal = None
        self.output_playing = False
        self.output_current_timer = QTimer(self)
        self.output_current_timer.timeout.connect(self.update_output_current_timer)
        self.output_current_time = 0

        self.frequencies = None
        self.original_fourier_transform = None
        self.fourier_transform = None
        self.last_index = 0
        self.current_timer = QTimer(self)
        self.current_timer.timeout.connect(self.update_current_timer)
        self.current_time = 0
        self.vertical_layout = QVBoxLayout()
        self.window_type = WindowType.RECTANGLE
        self.mode = ModeType.ANIMALS
        self.slider_values = []
        self._initialize_signals_slots()

    def _initialize_signals_slots(self):
        self.import_action.triggered.connect(self._import_signal_file)
        self.input_play_button.pressed.connect(self.play_time_signal)
        self.output_play_button.pressed.connect(self.play_time_output)
        self.input_slider.valueChanged.connect(self._on_input_slider_change)
        self.output_slider.valueChanged.connect(self._on_output_slider_change)
        self.rectangle_button.pressed.connect(self.rectangle)
        self.gaussian_button.pressed.connect(self.gaussian)
        self.hamming_button.pressed.connect(self.hamming)
        self.hanning_button.pressed.connect(self.hanning)
        self.uniform_range_action.triggered.connect(self.uniform_range_mode)
        self.musical_instruments_action.triggered.connect(self.music_mode)
        self.animal_sounds_action.triggered.connect(self.animals_mode)
        self.rectangle()
        self.animals_mode()
        


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
        self.generate_output_signal()

    def plot_input_frequency(self):
        self.frequencies, self.fourier_transform = self.apply_fourier_transform()
        self.original_fourier_transform = self.fourier_transform.copy()
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

    def slider_value_changed(self, index, value):
        self.slider_values[index].setText(f"{value}")
        if self.signal:
            if self.window_type == WindowType.GAUSSIAN:
                self.perform_gaussian()
            elif self.window_type == WindowType.RECTANGLE:
                self.perform_rect()    
            elif self.window_type == WindowType.HAMMING:
                self.perform_hamming()    
            elif self.window_type == WindowType.HANNING:
               self.perform_hanning()    
        
        
    def uniform_range_mode(self):
        self.mode = ModeType.UNIFORM
        for i in range(10):
            new_vertical_layout = QVBoxLayout()
            label = QLabel(f'Range {i+1}')
            slider = QSlider()
            slider.setRange(0,10)
            slider.setValue(1)
            value_label = QLabel('1')
            new_vertical_layout.addWidget(label)
            new_vertical_layout.addWidget(slider)
            new_vertical_layout.addWidget(value_label)
            self.slider_values.append(value_label)
            self.sliders_layout.addLayout(new_vertical_layout)
            slider.valueChanged.connect(partial(self.slider_value_changed, i))

    def animals_mode(self):
        self.mode = ModeType.ANIMALS
        for i in range(4):
            new_vertical_layout = QVBoxLayout()
            if i == 0:
                label = QLabel('Horse')
            elif i == 1:  
                label = QLabel('Lion') 
            elif i == 2:  
                label = QLabel('Bee') 
            elif i == 3:  
                label = QLabel('Elephants') 
            slider = QSlider()
            slider.setRange(0,10)
            slider.setValue(1)
            value_label = QLabel('1')
            self.slider_values.append(value_label)
            new_vertical_layout.addWidget(label)
            new_vertical_layout.addWidget(slider)
            new_vertical_layout.addWidget(value_label)
            self.sliders_layout.addLayout(new_vertical_layout)
            slider.valueChanged.connect(partial(self.slider_value_changed, i))

    def music_mode(self):
        self.mode = ModeType.MUSIC
        for i in range(4):
            new_vertical_layout = QVBoxLayout()
            if i == 0:
                label = QLabel('Music 1')
            elif i == 1:  
                label = QLabel('Music 2') 
            elif i == 2:  
                label = QLabel('Music 3') 
            elif i == 3:  
                label = QLabel('Music 4') 
            slider = QSlider()
            slider.setRange(0,10)
            slider.setValue(1)
            value_label = QLabel('1')
            self.slider_values.append(value_label)
            new_vertical_layout.addWidget(label)
            new_vertical_layout.addWidget(slider)
            new_vertical_layout.addWidget(value_label)
            self.sliders_layout.addLayout(new_vertical_layout)
            slider.valueChanged.connect(partial(self.slider_value_changed, i))


    def gaussian(self):
        self.window_type = WindowType.GAUSSIAN
        self.gaussian_button.setStyleSheet("QPushButton { border: 2px solid #FFFFFF; }")
        self.hamming_button.setStyleSheet("")
        self.rectangle_button.setStyleSheet("")
        self.hanning_button.setStyleSheet("")

    def rectangle(self):
        self.window_type = WindowType.RECTANGLE
        self.gaussian_button.setStyleSheet("")
        self.hamming_button.setStyleSheet("")
        self.rectangle_button.setStyleSheet("QPushButton { border: 2px solid #FFFFFF; }")
        self.hanning_button.setStyleSheet("")

    def hamming(self):
        self.window_type = WindowType.HAMMING
        self.gaussian_button.setStyleSheet("")
        self.hamming_button.setStyleSheet("QPushButton { border: 2px solid #FFFFFF; }")
        self.rectangle_button.setStyleSheet("")
        self.hanning_button.setStyleSheet("")

    def hanning(self):
        self.window_type = WindowType.HANNING
        self.gaussian_button.setStyleSheet("")
        self.hamming_button.setStyleSheet("")
        self.rectangle_button.setStyleSheet("")
        self.hanning_button.setStyleSheet("QPushButton { border: 2px solid #FFFFFF; }")

    def perform_rect(self): 
        total = len(self.slider_values)
        result = self.original_fourier_transform
        all_wave = np.array([])
        for i in range(total):  
            lower_freq = i * (self.frequencies[-1] / total)
            upper_freq = (i + 1) * (self.frequencies[-1] / total)
            amplitude = int(self.slider_values[i].text())
            freq_range_mask = (self.frequencies >= lower_freq) & (self.frequencies <= upper_freq)
            rectangular_wave = np.ones(np.sum(freq_range_mask)) * amplitude * 100
            full_rectangular_wave = np.ones(len(self.frequencies)) * amplitude
            all_wave = np.concatenate((all_wave, rectangular_wave))
            result = np.where(freq_range_mask, result * full_rectangular_wave, result)


        self.fourier_transform = result
        self.frequency_graph.clear()
        self.frequency_graph.plot(self.frequencies, abs(self.original_fourier_transform.real))
        pen_c = pg.mkPen(color=(255, 0, 0))
        self.frequency_graph.plot(self.frequencies,all_wave,pen= pen_c)
        self.generate_output_signal() 

    def perform_gaussian(self):
        total = len(self.slider_values)
        result = self.original_fourier_transform
        all_wave = np.array([])
        for i in range(total):
            lower_freq = i * (self.frequencies[-1] / total)
            upper_freq = (i + 1) * (self.frequencies[-1] / total)
            amplitude = int(self.slider_values[i].text())
            freq_range_mask = self.frequencies[(self.frequencies >= lower_freq) & (self.frequencies <= upper_freq)]
            fourier_transform_mask = self.fourier_transform[(self.frequencies >= lower_freq) & (self.frequencies <= upper_freq)]
            
            mean = (upper_freq - lower_freq)/2 
            std_dev = 500  
            gaus_signal = gaussian(len(fourier_transform_mask), std_dev) * amplitude
            
            
            # gaussian_signal = (np.exp(-(freq_range_mask - mean)**2 / (2 * std_dev**2)) ) * amplitude
            # full_gaussian_signal = (np.exp(-(self.frequencies - mean)**2 / (2 * std_dev**2)) ) * amplitude
            result = np.where(freq_range_mask, fourier_transform_mask * gaus_signal, fourier_transform_mask)
            all_wave = np.concatenate((all_wave, result))


        self.fourier_transform = all_wave
        self.frequency_graph.clear()
        self.frequency_graph.plot(self.frequencies, abs(self.original_fourier_transform.real))
        pen_c = pg.mkPen(color=(255, 0, 0))
        self.frequency_graph.plot(self.frequencies,all_wave.real*1000,pen= pen_c)
        self.generate_output_signal()


        # lower_freq = index * (self.frequencies[-1] / total)
        # upper_freq = (index + 1) * (self.frequencies[-1] / total)

        # # Create a Gaussian signal



        # # Create a frequency range mask
        # freq_range_mask = (self.frequencies >= lower_freq) & (self.frequencies <= upper_freq)
        # mean = (upper_freq - lower_freq)/2 
        # std_dev = 500  
        # gaussian_signal = (np.exp(-(self.frequencies - mean)**2 / (2 * std_dev**2)) ) * amplitude

        
        # result = np.where(freq_range_mask, self.fourier_transform * gaussian_signal, self.fourier_transform)

        # # self.fourier_transform = result
        # self.frequency_graph.clear()
        # self.frequency_graph.plot(self.frequencies, abs(result))
        # self.generate_output_signal()

    def perform_hanning(self):
        
        total = len(self.slider_values)
        result = self.original_fourier_transform
        all_wave = np.array([])
        for i in range(total):
            lower_freq = i * (self.frequencies[-1] / total)
            upper_freq = (i + 1) * (self.frequencies[-1] / total)
            amplitude = int(self.slider_values[i].text())
            freq_range_mask = self.frequencies[(self.frequencies >= lower_freq) & (self.frequencies <= upper_freq)]
            fourier_transform_mask = self.fourier_transform[(self.frequencies >= lower_freq) & (self.frequencies <= upper_freq)]
            
            mean = (upper_freq - lower_freq)/2 
            std_dev = 500  
            hanning_signal = np.hanning(len(fourier_transform_mask)) * amplitude
            
            
            # gaussian_signal = (np.exp(-(freq_range_mask - mean)**2 / (2 * std_dev**2)) ) * amplitude
            # full_gaussian_signal = (np.exp(-(self.frequencies - mean)**2 / (2 * std_dev**2)) ) * amplitude
            result = np.where(freq_range_mask, fourier_transform_mask * hanning_signal, fourier_transform_mask)
            all_wave = np.concatenate((all_wave, result))


        self.fourier_transform = all_wave
        self.frequency_graph.clear()
        self.frequency_graph.plot(self.frequencies, abs(self.original_fourier_transform.real))
        pen_c = pg.mkPen(color=(255, 0, 0))
        self.frequency_graph.plot(self.frequencies,all_wave.real,pen= pen_c)
        self.generate_output_signal()
        # lower_freq = 0
        # upper_freq = 5000
        # amplitude = 100

        # # Create a Hanning window
        # hanning_window = np.hanning(len(self.fourier_transform)) * amplitude
        

        # # Create a frequency range mask
        # freq_range_mask = (self.frequencies >= lower_freq) & (self.frequencies <= upper_freq)

        
        # result = np.where(freq_range_mask, self.fourier_transform * hanning_window, self.fourier_transform)

        # # Update self.fourier_transform
        # self.fourier_transform = result

        # self.frequency_graph.plot(self.frequencies, abs(self.fourier_transform))
        # self.generate_output_signal()

    def perform_hamming(self):
        lower_freq = 0
        upper_freq = 5000
        amplitude = 100
        # Create a Hamming window
        hamming_window = np.hamming(len(self.fourier_transform)) * amplitude

        # Create a frequency range mask
        freq_range_mask = (self.frequencies >= lower_freq) & (self.frequencies <= upper_freq)

        
        result = np.where(freq_range_mask, self.fourier_transform * hamming_window, self.fourier_transform)

        # Update self.fourier_transform
        self.fourier_transform = result

        self.frequency_graph.plot(self.frequencies, abs(self.fourier_transform))
        self.generate_output_signal()

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
