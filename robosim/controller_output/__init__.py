import enum
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

from robosim.simulator.data_types import ConfigurationSpaceType

if TYPE_CHECKING:
    from robosim.simulator.robot_simulator import Robot


class ControllerJointType(enum.Enum):
    joint_index = enum.auto()
    joint_name = enum.auto()


class AbstractController(ABC):
    controller_input_type: ControllerJointType
    robot: "Robot"

    def set_robot(self, robot: "Robot"):
        self.robot = robot

    def set_qs(
        self,
        qs: ConfigurationSpaceType,
        joint_indexes: List[int] = None,
        joint_names: List[str] = None,
    ):
        assert not all(
            _input is not None for _input in (joint_indexes, joint_names)
        ), "Please only specify one of joint indexes or names"

        if self.controller_input_type is ControllerJointType.joint_name:
            if joint_names is None:
                joint_names = self.robot.target_joint_names
            self.set_qs_with_names(qs=qs, joint_names=joint_names)
        elif self.controller_input_type is ControllerJointType.joint_index:
            if joint_indexes is None:
                if joint_names is None:
                    joint_names = self.robot.target_joint_names
                joint_indexes = self.robot.joint_name_to_indexes(joint_names)
            if len(qs) != len(joint_indexes):
                print(
                    f"qs has length {len(qs)} but joint index has length {len(joint_indexes)}"
                )
            self.set_qs_with_indexes(qs=qs, joint_indexes=joint_indexes)
        else:
            raise NotImplementedError()

    def set_qs_with_indexes(self, qs: ConfigurationSpaceType, joint_indexes: List[int]):
        raise NotImplementedError()

    def set_qs_with_names(self, qs: ConfigurationSpaceType, joint_names: List[str]):
        raise NotImplementedError()
