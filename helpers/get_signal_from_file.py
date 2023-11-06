
from PyQt6 import QtCore, QtWidgets
from managers.signal_loader import ISignalLoader, TextSignalLoader, CSVSignalLoader, ExcelXSignalLoader, ExcelSignalLoader, Mp3SignalLoader, WavSignalLoader
from models.signal import Signal


def get_signal_from_file(app):
        # get path of signal files only of types (xls, csv, txt)
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(app, 'Single File', QtCore.QDir.rootPath(), "(*.csv);;(*.txt);;(*.xls);;(*.xlsx);;(*.mp3);;(*.wav)")
        if not file_path:
            return None
        # check the type of signal file
        file_type = file_path.split('.')[-1]

        # Picking the right loader from file_type
        loader: ISignalLoader
        if file_type == 'xls':
            loader = ExcelSignalLoader()
        elif file_type == 'xlsx':
            loader = ExcelXSignalLoader()
        elif file_type == 'csv':
            loader = CSVSignalLoader()
        elif file_type == 'txt':
            loader = TextSignalLoader()
        elif file_type == 'mp3':
            loader = Mp3SignalLoader()
        else:
            loader = WavSignalLoader()
            
        
        signal: Signal = loader.load(file_path)
        return signal