"""
This class refer to the right area in the plots, with the 4 graphs, which are actualized in real time.
"""

import pyqtgraph as pg



class PlotPanel:

    def __init__(self, network, recorder):

        self.network = network
        self.recorder = recorder

        self.create_widget()
        self.create_layout()
        self.create_plots()

    def create_widget(self):

        self.widget = pg.GraphicsLayoutWidget()

    def create_layout(self):

        self.layout = self.widget.ci

    def create_plots(self):

        self.create_voltage_plot()
        self.create_astro_plot()
        self.create_raster_plot()
        self.create_rate_plot()

    def create_voltage_plot(self):
        self.voltage_plot = self.widget.addPlot(title="Selected Neuron Voltage")
        self.voltage_plot.showGrid(x=True, y=True)
        self.voltage_curve = self.voltage_plot.plot(pen=pg.mkPen((255, 180, 0), width=2)) 
        self.widget.nextRow()

    def create_astro_plot(self):
        self.astro_plot = self.widget.addPlot(title="Selected Astrocyte")
        self.astro_plot.showGrid(x=True, y=True)
        self.astro_curve = self.astro_plot.plot(pen=pg.mkPen((0, 220, 255), width=2)) 
        self.widget.nextRow()

    def create_raster_plot(self):

        self.raster_plot = self.widget.addPlot(title="Spike Raster")
        self.raster_plot.showGrid(x=True, y=True)

        self.raster_plot.setLabel("left", "Neuron")
        self.raster_plot.setLabel("bottom", "Time (ms)")

        self.raster_scatter = pg.ScatterPlotItem(size=5, brush=pg.mkBrush(255, 255, 255))
        self.raster_plot.addItem(self.raster_scatter)

        self.widget.nextRow()

    def create_rate_plot(self):

        self.rate_plot = self.widget.addPlot(title="Population Firing Rate")
        self.rate_plot.showGrid(x=True, y=True)

        self.rate_plot.setLabel("left", "Rate (Hz)")
        self.rate_plot.setLabel("bottom", "Time (ms)")

        self.rate_curve = self.rate_plot.plot(pen=pg.mkPen((255, 80, 80), width=2))