# Signal Equalizer Studio
![Image-Equalizer]()

Signal Equalizer Studio is a versatile desktop application designed for use in the music and speech industry, as well as various biomedical applications such as hearing aid abnormalities detection. The application allows users to open a signal and manipulate the magnitude of specific frequency components through intuitive sliders, offering a range of modes to suit different use cases.

## Modes

### 1. Uniform Range Mode
In this mode, the total frequency range of the input signal is divided uniformly into 10 equal frequency ranges. Each range is controlled by a dedicated slider in the user interface (UI). The application supports validation through the use of synthetic signals, enabling users to track the impact of equalizer actions on individual frequencies.

### 2. Musical Instruments Mode
This mode enables users to control the magnitude of specific musical instruments within an input music signal. The signal is a mixture of at least four different musical instruments, and each slider corresponds to a particular instrument, allowing for fine-tuning.

### 3. Animal Sounds Mode
Users can manipulate the magnitude of specific animal sounds within a mixture of at least four animal sounds. Each slider is associated with a particular animal sound, providing control over the composition of the output signal.

### 4. ECG Abnormalities Mode
For biomedical applications, the application supports ECG signals with abnormalities. Users can choose from four different ECG signals, including one normal signal and three signals with specific types of arrhythmias. Sliders in this mode control the magnitude of the arrhythmia component in the input signal.

## Frequency Range Manipulation
The application incorporates a multiplication/smoothing window for each frequency range controlled by the sliders. Users can choose from four available windows (Rectangle, Hamming, Hanning, Gaussian) to apply to the equalizer. The UI provides options to customize the parameters of the chosen window, visualize the customization, and apply it to the equalizer.

## User Interface
The UI is designed for ease of use and consistency across modes. Users can switch between modes seamlessly, with changes in slider captions and potentially the number of sliders being the main variations. Key UI components include:

- **Sliders:** Intuitive controls for manipulating frequency components.
- **Cine Signal Viewers:** Two linked cine signal viewers (input and output) with comprehensive functionality panels, enabling users to play, stop, pause, control speed, zoom, pan, and reset. The viewers are precisely linked to display the same time-part of the signal synchronously.
- **Spectrograms:** Two spectrograms (input and output) provide visualizations of the signal's frequency content. Any changes in the equalizer sliders are immediately reflected in the output spectrogram.
- **Toggle Option:** Users can toggle the visibility of the spectrograms based on preference.

## Getting Started

1. **Installation:**
   - Clone the repository.
   - Install the required dependencies.

2. **Run the Application:**
   - Launch the Signal Equalizer application.

3. **Select Mode:**
   - Choose the desired mode from the option menu or combobox.

4. **Adjust Sliders:**
   - Fine-tune the output by adjusting sliders based on the selected mode.

5. **Visualize Results:**
   - Use cine viewers and spectrograms to observe changes in the input and output signals.
