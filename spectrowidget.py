from PyQt6.QtWidgets import*
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor


from matplotlib.backends.backend_qt5agg import FigureCanvas

from matplotlib.figure import Figure


class SpectroWidget(QWidget):
    
    def __init__(self, parent = None):
        
        QWidget.__init__(self, parent)
        self.canvas = FigureCanvas(Figure())
        
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        
        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.canvas.figure.patch.set_facecolor('none')
        self.canvas.axes.tick_params(axis='x', colors='white')
        self.canvas.axes.tick_params(axis='y', colors='white')
        self.canvas.setStyleSheet("background-color:black;")
        self.setLayout(vertical_layout)