from typing import List

import pybullet_tools.utils as pu

from robosim.controller_output import AbstractController
from robosim.simulator.data_types import ConfigurationSpaceType
from robosim.controller_output import ControllerJointType, AbstractController
from robosim.simulator.robot_simulator import Robot


class MultiCastController(AbstractController):
    controller_input_type: ControllerJointType = ControllerJointType.joint_index

    def __init__(self, controllers: List[AbstractController]) -> None:
        super().__init__()
        self.controllers = controllers

    def set_robot(self, robot: Robot):
        for controller in self.controllers:
            controller.set_robot(robot)

    def set_qs(
        self,
        qs: ConfigurationSpaceType,
        joint_indexes: List[int] = None,
        joint_names: List[str] = None,
    ):
        for controller in self.controllers:
            controller.set_qs(
                qs=qs, joint_indexes=joint_indexes, joint_names=joint_names
            )
