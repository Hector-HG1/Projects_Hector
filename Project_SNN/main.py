"""
This file calll the others, it create the network, call the recorder, initialize and run the view, and start the simulation.
If you want to modifie the number of neurons and astrocyte it's here, others parameters are in the config.py and equations.py files.
"""

from NETWORK.Network import SNN_Network
from DATA.DataRecorder import DataRecorder
from VIEW.MainView import MainView
from SIMULATION.Simulation import Simulation
from NETWORK.Config import NORMAL_PARAMS

def main():

    network = SNN_Network(
        N_NEURONS = 100,
        N_ASTRO = 10,
        CONNECTION_PROB = NORMAL_PARAMS['connection_prob'],
        params = NORMAL_PARAMS,
    )

    recorder = DataRecorder(network)

    view = MainView(network, recorder)

    sim = Simulation(network, recorder, view)
    sim.start()

    view.show()


if __name__ == '__main__':
    main()