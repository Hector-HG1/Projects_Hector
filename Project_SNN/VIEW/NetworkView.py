"""
This class refer to the network view, where you see the network in the middle area. 
it generate all the "animation".
"""

import numpy as np
import pyqtgraph as pg


class NetworkView:

    def __init__(self, network, recorder):

        self.network = network
        self.recorder = recorder

        self.create_widget()
        self.create_plot()
        self.create_connections()
        self.create_node_layers()

    def create_widget(self):
        self.widget = pg.GraphicsLayoutWidget()

    def create_plot(self):
        self.plot = self.widget.addPlot()
        self.plot.setAspectLocked(True)
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")
        self.plot.setXRange(-1.5, 1.5)
        self.plot.setYRange(-1.5, 1.5)

    def create_node_layers(self):
        self.neuron_scatter = pg.ScatterPlotItem()
        self.astro_scatter = pg.ScatterPlotItem()
        self.plot.addItem(self.neuron_scatter)
        self.plot.addItem(self.astro_scatter)
        self.neuron_core = self.neuron_scatter
        self.astro_core = self.astro_scatter

    def create_connections(self):
        self.connection_lines = []
        for i, j in zip(self.network.S.i[:], self.network.S.j[:]):
            x0, y0 = self.network.neuron_positions[int(i)]
            x1, y1 = self.network.neuron_positions[int(j)]
            line = pg.PlotCurveItem(
                [x0, x1], [y0, y1],
                pen=pg.mkPen((100, 100, 100, 80), width=1)
            )
            self.plot.addItem(line)
            self.connection_lines.append(line)
        self.create_astro_connections()

    def create_astro_connections(self):
        self.astro_lines = []
        for astro_idx, neuron_list in self.network.astro_to_neuron.items():
            ax, ay = self.network.astro_positions[astro_idx]
            for n in neuron_list:
                nx, ny = self.network.neuron_positions[n]
                line = pg.PlotCurveItem(
                    [ax, nx], [ay, ny],
                    pen=pg.mkPen((0, 180, 255, 50), width=1)
                )
                self.plot.addItem(line)
                self.astro_lines.append(line)