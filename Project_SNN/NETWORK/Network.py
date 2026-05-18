"""
this contains all the informations and operations to create the network when it is initialized. It call the config and the equations files, so parametring is not mainly in this class.
It refer in the documentation as the programation of the neurons synapses astrocytes and network in the section mathematics and biology. 
"""

from brian2 import *
from brian2.utils.logger import BrianLogger
from NETWORK.equations import NEURON_EQUATION, ASTRO_EQUATION, SYNAPSE_EQUATION, SYNAPSE_ON_PRE
from NETWORK.Config import NORMAL_PARAMS
import numpy as np


class SNN_Network:

    def __init__(self, N_NEURONS, N_ASTRO, CONNECTION_PROB,
                 params=None):
        self.N_NEURONS = N_NEURONS
        self.N_ASTRO = N_ASTRO
        self.CONNECTION_PROB = CONNECTION_PROB
        self.params = params if params is not None else NORMAL_PARAMS

        start_scope()
        defaultclock.dt = 0.01 * ms

        BrianLogger.suppress_hierarchy('brian2.codegen.generators.base')

        self._rng = np.random.default_rng()

        self.create_neurons()
        self.create_astrocytes()
        self.create_synapses()
        self.create_network_operations()
        self.create_monitors()
        self.create_neuron_position()
        self.create_astro_position()
        self.create_mapping()
        self.create_stimulation()
        self.create_network()

    def create_neurons(self):
        p = self.params
        N = self.N_NEURONS
        rng = self._rng

        self.G = NeuronGroup(
            N, NEURON_EQUATION,
            threshold = 'v > -20*mV',
            refractory = 'v > -20*mV',
            method = 'rk4',
            namespace = {'tau_gsyn': p['tau_gsyn']},
        )

        self.G.v = (p['v0'] / mV + rng.uniform(-5, 5, N)) * mV
        self.G.m = np.clip(rng.normal(p['m0'], 0.02, N), 0, 1)
        self.G.h = np.clip(rng.normal(p['h0'], 0.05, N), 0, 1)
        self.G.n = np.clip(rng.normal(p['n0'], 0.02, N), 0, 1)

        self.G.gsyn = 0 * msiemens/cm**2
        self.G.Esyn = 0 * mV
        self.G.Iext = 0 * uA/cm**2

        self._bg_arr = np.clip(p['I_bg'] * (1 + rng.normal(0, 0.20, N)), 0, None)
        self._sigma_arr = np.clip(p['I_noise'] * (1 + rng.normal(0, 0.30, N)), 0, None)

    def create_astrocytes(self):
        p = self.params
        self.A = NeuronGroup(
            self.N_ASTRO, ASTRO_EQUATION,
            method = 'euler',
            namespace = {'tau_A': p['tau_A']},
        )
        self.A.A = 0
        self.A.coupling_input = 0

    def create_synapses(self):
        p = self.params
        rng = self._rng
        n_inh = max(1, int(round(self.N_NEURONS * 0.20)))
        inh_idx = rng.choice(self.N_NEURONS, size=n_inh, replace=False)
        self.is_inhibitory = np.zeros(self.N_NEURONS, dtype=bool)
        self.is_inhibitory[inh_idx] = True

        self.S = Synapses(
            self.G, self.G,
            model = SYNAPSE_EQUATION,
            on_pre = SYNAPSE_ON_PRE,
            namespace = {'alpha': p['alpha']},
            delay = 1 * ms,
        )
        self.S.connect(condition='i != j', p=self.CONNECTION_PROB)
        if len(self.S.i) == 0:
            self.S.connect(i=0, j=1)

        self.S.astro_val = 0

        pre_arr = np.array(self.S.i[:], dtype=int)
        inh_mask = self.is_inhibitory[pre_arr]

        w0_exc_f = float(p['w0_exc'] / (msiemens/cm**2))
        w0_inh_f = float(p['w0_inh'] / (msiemens/cm**2))
        Esyn_exc_f = float(p['Esyn_exc'] / mV)
        Esyn_inh_f = float(p['Esyn_inh'] / mV)

        self.S.w0[:] = np.where(inh_mask, w0_inh_f, w0_exc_f) * msiemens/cm**2
        self.S.Esyn_s[:] = np.where(inh_mask, Esyn_inh_f, Esyn_exc_f) * mV
        self.S.w_eff[:] = self.S.w0[:]

    def create_network_operations(self):
        G = self.G
        A = self.A
        S = self.S
        bg_arr = self._bg_arr
        sigma_arr = self._sigma_arr
        rng = self._rng
        p = self.params

        @network_operation(dt=1*ms)
        def update_astro():
            noise = rng.standard_normal(len(G.v)) * sigma_arr
            G.Iext = (bg_arr + noise) * uA/cm**2
            gsyn_arr = np.array(G.gsyn[:] / (msiemens/cm**2))
            for astro_idx, neuron_list in self.astro_to_neuron.items():
                A.coupling_input[astro_idx] = float(np.sum(gsyn_arr[neuron_list]))
            S_j_arr = np.array(S.j[:], dtype=int)
            alpha_v = p['alpha']
            for astro_idx, neuron_list in self.astro_to_neuron.items():
                mask = np.isin(S_j_arr, neuron_list)
                a_val = float(A.A[astro_idx])
                S.astro_val[mask] = a_val
                w0_arr = np.array(S.w0[mask] / (msiemens/cm**2))
                S.w_eff[mask] = w0_arr * (1.0 + alpha_v * a_val) * msiemens/cm**2

        self._update_astro = update_astro

    def create_monitors(self):
        self.spike_monitor = SpikeMonitor(self.G)

    def create_neuron_position(self):
        self.neuron_positions = self._rng.uniform(
            low=-1, high=1, size=(self.N_NEURONS, 2)
        )

    def create_astro_position(self):
        self.astro_positions = self._rng.uniform(
            low=-0.7, high=0.7, size=(self.N_ASTRO, 2)
        )

    def create_mapping(self):
        self.astro_to_neuron = {}
        neuron_per_astro = max(1, self.N_NEURONS // self.N_ASTRO)
        all_neurons = np.arange(self.N_NEURONS)
        self._rng.shuffle(all_neurons)
        for a in range(self.N_ASTRO):
            start = a * neuron_per_astro
            end = min((a + 1) * neuron_per_astro, self.N_NEURONS)
            self.astro_to_neuron[a] = list(all_neurons[start:end])

    def create_stimulation(self):
        self.pulse_remaining = np.zeros(self.N_NEURONS)
        self.warmup_spike_count = 0

    def create_network(self):
        self.brian_network = Network(
            self.G, self.A, self.S,
            self.spike_monitor,
            self._update_astro,
        )
        self.G.Iext = 0 * uA/cm**2
        self.brian_network.run(100 * ms, namespace=self.get_namespace())
        self.warmup_spike_count = len(self.spike_monitor.t)

    def get_namespace(self):
        p = self.params
        return {
            'gNa': p['gNa'],
            'gK': p['gK'],
            'gL': p['gL'],
            'ENa': p['ENa'],
            'EK': p['EK'],
            'El': p['El'],
            'Cm': p['Cm'],
            'tau_gsyn': p['tau_gsyn'],
        }