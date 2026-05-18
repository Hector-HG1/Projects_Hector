"""
This file store the equations, there are the same for the two configurations, all the explanations are in the documentation, see "Mathematics and biology". 
"""

from brian2 import *

NEURON_EQUATION = """
dv/dt = (Iext + I_bg_neuron - INa - Ik - Il + Isyn) / Cm : volt

INa = gNa * m**3 * h * (v - ENa) : amp/meter**2
Ik = gK * n**4 * (v - EK): amp/meter**2
Il = gL * (v - El): amp/meter**2

Isyn = gsyn * (Esyn - v) : amp/meter**2
Esyn : volt

dgsyn/dt = -gsyn / tau_gsyn : siemens/meter**2

dm/dt = alpham * (1 - m) - betam * m : 1
dh/dt = alphah * (1 - h) - betah * h : 1
dn/dt = alphan * (1 - n) - betan * n : 1

alpham = (0.1/mV) * 10*mV / exprel(-(v+40*mV)/(10*mV)) / ms : Hz
betam = 4 * exp(-(v + 65*mV) / (18*mV)) / ms : Hz

alphah = 0.07 * exp(-(v + 65*mV) / (20*mV)) / ms : Hz
betah = 1 / (1 + exp(-(v + 35*mV) / (10*mV))) / ms : Hz

alphan = (0.01/mV) * 10*mV / exprel(-(v+55*mV)/(10*mV)) / ms : Hz
betan = 0.125 * exp(-(v + 65*mV) / (80*mV)) / ms : Hz

I_bg_neuron : amp/meter**2
Iext : amp/meter**2
"""

ASTRO_EQUATION = """
dA/dt = (-A + coupling_input) / tau_A : 1
coupling_input : 1
"""

SYNAPSE_EQUATION = """
w0 : siemens/meter**2
astro_val : 1
Esyn_s : volt
w_eff : siemens/meter**2
"""

SYNAPSE_ON_PRE = """
gsyn_post += w_eff
Esyn_post = Esyn_s
"""