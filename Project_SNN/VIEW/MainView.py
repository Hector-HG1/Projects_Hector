"""
This class create the main window, and create the three compartiments, organize and call them
"""

from pyqtgraph.Qt import QtWidgets

from VIEW.PlotPanel import PlotPanel
from VIEW.NetworkView import NetworkView
from VIEW.ControlPanel import ControlPanel


class MainView:

    def __init__(self, network, recorder):

        self.network = network
        self.recorder = recorder

        self.create_window()
        self.create_layout()
        self.create_subviews()

    def create_window(self):
        self.app = QtWidgets.QApplication([])
        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle("SNN Network Simulation")
        self.window.resize(1600, 900)

    def create_layout(self):
        self.main_layout = QtWidgets.QHBoxLayout()
        self.window.setLayout(self.main_layout)

    def create_subviews(self):
        self.plot_panel = PlotPanel(self.network, self.recorder)
        self.network_view = NetworkView(self.network, self.recorder)
        self.control_panel = ControlPanel(self.network, self.recorder)

        self.main_layout.addLayout(self.control_panel.layout, 1)
        self.main_layout.addWidget(self.network_view.widget, 2)
        self.main_layout.addWidget(self.plot_panel.widget, 3)

    def show(self):
        self.window.show()
        self.app.exec()