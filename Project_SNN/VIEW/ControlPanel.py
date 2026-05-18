"""
This class refer to the left side of the GUI, with the button : start / pause , normal / epileptic , reset and export to csv. 
the utilisation of this HMI is descripted in the user guide of the documentation. 
"""

from pyqtgraph.Qt import QtWidgets, QtCore


class ControlPanel:

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"

    def __init__(self, network, recorder):
        self.network = network
        self.recorder = recorder
        self._state = self.IDLE
        self._has_data = False 

        self.create_layout()
        self.create_mode_group()
        self.create_run_buttons()
        self.create_export_button()
        self.create_reset_button()
        self.create_separator()
        self.create_debug_info()

        self.layout.addStretch()
        self._refresh_button_states()

    def create_layout(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setSpacing(10)
        self.layout.setContentsMargins(12, 12, 12, 12)

    def create_mode_group(self):
        group_box = QtWidgets.QGroupBox("Mode")
        group_layout = QtWidgets.QHBoxLayout()

        self.btn_normal = QtWidgets.QPushButton("Normal")
        self.btn_epileptic = QtWidgets.QPushButton("Epileptic")

        self.btn_normal.setCheckable(True)
        self.btn_epileptic.setCheckable(True)
        self.btn_normal.setChecked(True) 

        self.btn_normal.setToolTip("Classic Hodgkin-Huxley — only selectable before first run")
        self.btn_epileptic.setToolTip("Hyperexcitable network — only selectable before first run")

        group_layout.addWidget(self.btn_normal)
        group_layout.addWidget(self.btn_epileptic)
        group_box.setLayout(group_layout)

        self.layout.addWidget(group_box)
        self.mode_group_box = group_box

        self.btn_normal.clicked.connect(lambda: self._select_mode("normal"))
        self.btn_epileptic.clicked.connect(lambda: self._select_mode("epileptic"))

    def _select_mode(self, mode):
        if mode == "normal":
            self.btn_normal.setChecked(True)
            self.btn_epileptic.setChecked(False)
        else:
            self.btn_normal.setChecked(False)
            self.btn_epileptic.setChecked(True)

    def get_mode(self):
        return "epileptic" if self.btn_epileptic.isChecked() else "normal"

    def create_run_buttons(self):
        row = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("▶ Start")
        self.btn_pause = QtWidgets.QPushButton("⏸ Pause")
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_pause)
        self.layout.addLayout(row)

        self.btn_start.setToolTip("Start or resume the simulation")
        self.btn_pause.setToolTip("Pause the simulation")

    def create_export_button(self):
        self.btn_export = QtWidgets.QPushButton("⬇  Export Data")
        self.btn_export.setToolTip("Export recorded data to CSV — available after simulation has run")
        self.layout.addWidget(self.btn_export)

    def create_reset_button(self):
        self.btn_reset = QtWidgets.QPushButton("↺  Reset")
        self.btn_reset.setToolTip("Stop and rebuild the network from scratch")
        self.layout.addWidget(self.btn_reset)

    def create_separator(self):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.layout.addWidget(line)

    def create_debug_info(self):
        self.debug_label = QtWidgets.QLabel(
            "Selected neuron : –\nSelected astro : –"
        )
        self.debug_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self.layout.addWidget(self.debug_label)

    def set_state(self, state):
        self._state = state
        self._refresh_button_states()

    def notify_has_data(self):
        self._has_data = True
        self._refresh_button_states()

    def _refresh_button_states(self):
        s = self._state

        mode_enabled = (s == self.IDLE)
        self.btn_normal.setEnabled(mode_enabled)
        self.btn_epileptic.setEnabled(mode_enabled)
        self.mode_group_box.setEnabled(mode_enabled)

        self.btn_start.setEnabled(s in (self.IDLE, self.PAUSED))

        self.btn_pause.setEnabled(s == self.RUNNING)

        self.btn_export.setEnabled(
            self._has_data and s in (self.PAUSED, self.STOPPED)
        )

        self.btn_reset.setEnabled(True)

    def update_debug(self, neuron_idx, astro_idx):
        self.debug_label.setText(
            f"Selected neuron : {neuron_idx}\n"
            f"Selected astro : {astro_idx}"
        )