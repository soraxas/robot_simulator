from dataclasses import dataclass
from functools import cached_property
from typing import List, Dict

import quaternion

from .tf import Transform


@dataclass
class Pose:
    position: List[float]
    orientation: quaternion.quaternion

    def __init__(self, position, quat, order: str = "xyzw"):
        self.position = position
        if not isinstance(quat, quaternion.quaternion):
            if order == "xyzw":
                quat = quaternion.quaternion(quat[3], quat[0], quat[1], quat[2])
            elif order == "wxyz":
                quat = quaternion.quaternion(*quat)
        self.orientation = quat

    @cached_property
    def transformation_matrix(self):
        return Transform(pos=self.position, quat=self.orientation)

    def composite(self, other: "Pose"):
        trans = self.transformation_matrix.composition(other.transformation_matrix)
        return Pose(trans.position(), trans.quaternion())

    # def __add__(self, other: "Pose"):
    #     assert isinstance(other, Pose)
    #     assert len(self.position) == len(other.position)
    #     assert len(self.orientation) == len(other.orientation)
    #     return Pose(
    #         [self.position[i] + other.position[i] for i in range(len(self.position))],
    #         self.orientation * other.orientation,
    #     )

    def __iter__(self):
        yield self.position
        quat = self.orientation
        yield [quat.x, quat.y, quat.z, quat.w]


@dataclass
class JointState:
    name: List[str]
    position: List[float]

    @cached_property
    def name_to_index_mapping(self) -> Dict[str, int]:
        return {_name: i for i, _name in enumerate(self.name)}

    def get(self, joint_state_names: List[str]) -> List[float]:
        return [
            self.position[self.name_to_index_mapping[name]]
            for name in joint_state_names
        ]
