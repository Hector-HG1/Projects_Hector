"""
This class initialize all the variables need to be record, and record them. 
This values record will be plotted by plotpanel.py and can be used to export tham as csv.
"""

from brian2 import mV, ms
from collections import deque



class DataRecorder:

    def __init__(self, SNN_network, maxlen=2000):
        self.network = SNN_network
        self.maxlen = maxlen
        self.initialize_time()
        self.initialize_voltage_history()
        self.initialize_astro_history()
        self.initialize_raster()
        self.initialize_population_rate()
        self.last_spike_index = getattr(SNN_network, 'warmup_spike_count', 0)


    def initialize_time(self):
        self.time_data = deque(maxlen=self.maxlen)
        self.time_ms = 0

    def initialize_voltage_history(self):
        self.neuron_voltage_history = [
            deque(maxlen=self.maxlen) for _ in range(self.network.N_NEURONS)
        ]

    def initialize_astro_history(self):
        self.astro_history = [
            deque(maxlen=self.maxlen) for _ in range(self.network.N_ASTRO)
        ]

    def initialize_raster(self):
        self.raster_x = deque(maxlen=5000)
        self.raster_y = deque(maxlen=5000)

    def initialize_population_rate(self):
        self.rate_history = deque(maxlen=self.maxlen)
        self._rate_window_ms = 50
        self._rate_window_deque = deque()
        self._spike_times_ms = deque()

    def update_time(self, dt_ms):
        self.time_ms += dt_ms

    def record_time(self):
        self.time_data.append(self.time_ms)

    def record_voltage(self):
        for i in range(self.network.N_NEURONS):
            self.neuron_voltage_history[i].append(
                float(self.network.G.v[i] / mV)
            )

    def record_astro_activity(self):

        for i in range(self.network.N_ASTRO):
            self.astro_history[i].append(float(self.network.A.A[i]))

    def record_raster(self):
        spike_times = self.network.spike_monitor.t / ms
        spike_indices = self.network.spike_monitor.i

        new_times = spike_times[self.last_spike_index:]
        new_indices = spike_indices[self.last_spike_index:]

        for t, i in zip(new_times, new_indices):
            self.raster_x.append(self.time_ms)
            self.raster_y.append(int(i))
            self._spike_times_ms.append(self.time_ms)

        self.last_spike_index = len(spike_times)

    def record_population_rate(self):
        cutoff = self.time_ms - self._rate_window_ms
        while self._spike_times_ms and self._spike_times_ms[0] < cutoff:
            self._spike_times_ms.popleft()
        recent_spikes = len(self._spike_times_ms)
        rate_hz = recent_spikes / self.network.N_NEURONS / (self._rate_window_ms / 1000.0)
        self.rate_history.append(rate_hz)

    def record_step(self):
        self.record_time()
        self.record_voltage()
        self.record_astro_activity()
        self.record_raster()
        self.record_population_rate()