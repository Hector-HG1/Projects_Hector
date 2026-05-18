"""
This section refer in the doc to the Algorithm section, This is the section which call the logic and the GUI and update them both.
This file initialize the networks with the right mode 
"""

from NETWORK.Config import NORMAL_PARAMS, EPILEPTIC_PARAMS
from NETWORK.Network import SNN_Network
from DATA.DataRecorder import DataRecorder
from VIEW.NetworkView import NetworkView

from pyqtgraph.Qt import QtCore, QtWidgets
from brian2 import ms, mV, msiemens, cm
import pyqtgraph as pg
import numpy as np
import csv, os
from datetime import datetime


class Simulation:

    IDLE = 'IDLE'
    RUNNING = 'RUNNING'
    PAUSED = 'PAUSED'
    STOPPED = 'STOPPED'

    def __init__(self, network, recorder, view, dt=0.1):
        self.network = network
        self.recorder = recorder
        self.view = view
        self.dt = dt
        self.time_ms = 0
        self.neuron_flash = np.zeros(network.N_NEURONS)

        self.selected_neuron = network.N_NEURONS // 2
        self.selected_astro = 0
        self._state = self.IDLE

        self._connect_clicks()
        self._connect_buttons()
        self.create_timer()

    def _set_state(self, state):
        self._state = state
        self.view.control_panel.set_state(state)

    def _connect_buttons(self):
        cp = self.view.control_panel
        cp.btn_start.clicked.connect(self._on_start)
        cp.btn_pause.clicked.connect(self._on_pause)
        cp.btn_export.clicked.connect(self._on_export)
        cp.btn_reset.clicked.connect(self._on_reset)

    def _on_start(self):
        if self._state in (self.IDLE, self.PAUSED):
            self._set_state(self.RUNNING)
            self.timer.start(20)

    def _on_pause(self):
        if self._state == self.RUNNING:
            self.timer.stop()
            self._set_state(self.PAUSED)

    def _on_export(self):
        if self._state == self.RUNNING:
            return
        self._export_to_csv()

    def _on_reset(self):
        self.timer.stop()
        self._set_state(self.STOPPED)

        mode   = self.view.control_panel.get_mode()
        params = EPILEPTIC_PARAMS if mode == 'epileptic' else NORMAL_PARAMS

        self.network = SNN_Network(
            N_NEURONS = self.network.N_NEURONS,
            N_ASTRO = self.network.N_ASTRO,
            CONNECTION_PROB = params['connection_prob'],
            params = params,
        )
        self.recorder = DataRecorder(self.network)

        old_widget = self.view.network_view.widget
        self.view.network_view = NetworkView(self.network, self.recorder)
        layout = self.view.main_layout
        idx = layout.indexOf(old_widget)
        layout.removeWidget(old_widget)
        old_widget.setParent(None)
        layout.insertWidget(idx, self.view.network_view.widget, 2)

        self._connect_clicks()
        self.neuron_flash = np.zeros(self.network.N_NEURONS)
        self.selected_neuron = self.network.N_NEURONS // 2
        self.selected_astro = 0
        self.time_ms = 0

        self._set_state(self.IDLE)
        self._refresh_plot_titles()


    def _connect_clicks(self):
        self.view.network_view.neuron_core.sigClicked.connect(self._neuron_clicked)
        self.view.network_view.astro_core.sigClicked.connect(self._astro_clicked)

    def _neuron_clicked(self, plot_item, points):
        if len(points):
            idx = points[0].data()
            if idx is not None:
                self.selected_neuron = int(idx)
                self.network.pulse_remaining[int(idx)] = 5 
                self._refresh_plot_titles()

    def _astro_clicked(self, plot_item, points):
        if len(points):
            idx = points[0].data()
            if idx is not None:
                self.selected_astro = int(idx)
                self._refresh_plot_titles()

    def _refresh_plot_titles(self):
        panel = self.view.plot_panel
        ntype = 'INH' if self.network.is_inhibitory[self.selected_neuron] else 'EXC'
        panel.voltage_plot.setTitle(
            f'Neuron Voltage [N{self.selected_neuron} — {ntype}]'
        )
        panel.astro_plot.setTitle(
            f'Astrocyte Activity [A{self.selected_astro}]'
        )

    def create_timer(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.step)

    def start(self):
        self._refresh_plot_titles()
        self._set_state(self.IDLE)

    def step(self):
        if self._state != self.RUNNING:
            return
        self.time_ms += self.dt
        self.apply_stimulation()
        self.run_network()
        self.update_recorder()
        self.update_views()
        self.view.control_panel.notify_has_data()

    def apply_stimulation(self):
        unit = self.network.G.Iext.unit
        for i in range(self.network.N_NEURONS):
            if self.network.pulse_remaining[i] > 0:
                self.network.G.Iext[i] += 20 * unit
                self.network.pulse_remaining[i] -= 1

    def run_network(self):
        self.network.brian_network.run(
            self.dt * ms,
            namespace=self.network.get_namespace()
        )

    def update_recorder(self):
        self.recorder.update_time(self.dt)
        self.recorder.record_step()

    def update_views(self):
        self.update_plot_panel()
        self.update_network_view()
        self.update_control_panel()

    def update_plot_panel(self):
        panel = self.view.plot_panel
        rec = self.recorder
        x = list(rec.time_data)
        if not x:
            return

        panel.voltage_curve.setData(x, list(rec.neuron_voltage_history[self.selected_neuron]))
        panel.astro_curve.setData(x, list(rec.astro_history[self.selected_astro]))

        t_min, t_max = x[0], x[-1]
        rx = np.array(rec.raster_x)
        ry = np.array(rec.raster_y)
        if len(rx):
            mask = (rx >= t_min) & (rx <= t_max)
            panel.raster_scatter.setData(rx[mask].tolist(), ry[mask].tolist())
        else:
            panel.raster_scatter.setData([], [])

        rate_list = list(rec.rate_history)
        n = min(len(x), len(rate_list))
        panel.rate_curve.setData(x[-n:], rate_list[-n:])

    def update_network_view(self):
        view = self.view.network_view
        voltages_mV = np.array(self.network.G.v / mV)
        norm_v = np.clip((voltages_mV + 80) / 120, 0, 1)

        self.neuron_flash[voltages_mV > 0] = 1.0
        self.neuron_flash *= 0.92

        neuron_spots = []
        for i in range(self.network.N_NEURONS):
            f = self.neuron_flash[i]
            size = 14 + 22 * f + 8 * norm_v[i]
            neuron_spots.append({
                'pos': self.network.neuron_positions[i],
                'size': size,
                'brush': pg.mkBrush(255, int(140 + 115*f), int(200*f), int(80 + 175*f)),
                'pen': pg.mkPen(255, 255, 255, 180),
                'data': i,
            })
        view.neuron_scatter.setData(neuron_spots)

        astro_spots = []
        for i in range(self.network.N_ASTRO):
            strength = float(np.clip(self.network.A.A[i], 0, 1))
            astro_spots.append({
                'pos': self.network.astro_positions[i],
                'size': 16 + 30 * strength,
                'brush': pg.mkBrush(0, int(180 + 75*strength), 255, int(80 + 175*strength)),
                'pen': pg.mkPen(255, 255, 255, 180),
                'data': i,
            })
        view.astro_scatter.setData(astro_spots)

    def update_control_panel(self):
        self.view.control_panel.update_debug(self.selected_neuron, self.selected_astro)

    def _export_to_csv(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_name = f'snn_export_{timestamp}.csv'

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            None, 'Export simulation data', default_name,
            'CSV files (*.csv);;All files (*)'
        )
        if not filename:
            return
        if not filename.lower().endswith('.csv'):
            filename += '.csv'

        rec = self.recorder
        times = list(rec.time_data)
        n_rows = len(times)

        def pad(col):
            c = list(col)
            return [''] * (n_rows - len(c)) + c

        voltage_cols = [pad(h) for h in rec.neuron_voltage_history]
        astro_cols = [pad(h) for h in rec.astro_history]
        rate_col = pad(rec.rate_history)
        raster_t = list(rec.raster_x)
        raster_i = list(rec.raster_y)

        ntypes = ['I' if self.network.is_inhibitory[i] else 'E'
                  for i in range(self.network.N_NEURONS)]
        header = (
            ['time_ms']
            + [f'v_N{i}_{ntypes[i]}' for i in range(self.network.N_NEURONS)]
            + [f'A_A{i}' for i in range(self.network.N_ASTRO)]
            + ['rate_hz', 'spike_time_ms', 'spike_neuron']
        )

        n_total = max(n_rows, len(raster_t))
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for r in range(n_total):
                t = times[r] if r < n_rows else ''
                vs = [voltage_cols[n][r] for n in range(self.network.N_NEURONS)] \
                       if r < n_rows else [''] * self.network.N_NEURONS
                cas = [astro_cols[n][r] for n in range(self.network.N_ASTRO)] \
                       if r < n_rows else [''] * self.network.N_ASTRO
                rate = rate_col[r] if r < n_rows else ''
                st = raster_t[r] if r < len(raster_t) else ''
                si = raster_i[r] if r < len(raster_i) else ''
                writer.writerow([t] + vs + cas + [rate, st, si])

        QtWidgets.QMessageBox.information(
            None, 'Export complete', f'Data saved to:\n{filename}'
        )