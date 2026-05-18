"""
Here are stok all the values needed to initialize the neurons, astrocyte, synapse and network in the two configurations 
"""

from brian2 import msiemens, cm, mV, ufarad, ms


NORMAL_PARAMS = dict(

    gNa = 120 * msiemens/cm**2,
    gK = 36 * msiemens/cm**2,
    gL = 0.3 * msiemens/cm**2,
    ENa = 50 * mV,
    EK = -77 * mV,
    El = -54 * mV,
    Cm = 1 * ufarad/cm**2,

    v0 = -65 * mV,
    m0 = 0.05,
    h0 = 0.60,
    n0 = 0.32,

    I_bg = 0.5,
    I_noise = 0.2,
   
    tau_gsyn = 5 * ms,

    w0_exc = 0.10 * msiemens/cm**2,
    w0_inh = 0.15 * msiemens/cm**2,

    Esyn_exc = 0 * mV,
    Esyn_inh = -80 * mV,

    tau_A = 100 * ms,
    alpha = 0.2,

    connection_prob = 0.35,
)

EPILEPTIC_PARAMS = dict(

    gNa = 120 * msiemens/cm**2,
    gK = 27 * msiemens/cm**2, 
    gL = 0.20 * msiemens/cm**2,
    ENa = 50 * mV,
    EK = -77 * mV,
    El = -51 * mV,
    Cm = 1 * ufarad/cm**2,

    v0 = -65 * mV,
    m0 = 0.05,
    h0 = 0.60,
    n0 = 0.32,

    I_bg = 0.75,
    I_noise = 0.10,

    tau_gsyn = 7.5 * ms,

    w0_exc = 0.15 * msiemens/cm**2, 
    w0_inh = 0.09 * msiemens/cm**2,

    Esyn_exc = 0 * mV,
    Esyn_inh = -80 * mV,

    tau_A = 70 * ms, 
    alpha = 0.30,

    connection_prob = 0.50, 
)