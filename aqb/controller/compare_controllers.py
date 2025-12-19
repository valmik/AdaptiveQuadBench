import numpy as np
from aqb.controller.controller_template import MultirotorControlTemplate

class CompareControllers(MultirotorControlTemplate):
    def __init__(self, controllers):
        self.controllers = controllers

    def update(self, t, state, flat_output, trajectory=None):
        cmds = []
        for controller in self.controllers:
            cmds.append(controller.update(t, state, flat_output, trajectory=trajectory))

        # import pdb; pdb.set_trace()

        return cmds[0]