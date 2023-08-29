from typing import List

import pybullet_tools.utils as pu

from robosim.controller_output import AbstractController
from robosim.simulator.data_types import ConfigurationSpaceType
from robosim.controller_output import ControllerJointType, AbstractController


class PyBulletController(AbstractController):
    controller_input_type: ControllerJointType = ControllerJointType.joint_index

    def set_qs_with_indexes(self, qs: ConfigurationSpaceType, joint_indexes: List[int]):
        pu.set_joint_positions(self.robot.pyb_robot_id, joint_indexes, qs)
