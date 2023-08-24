from dataclasses import dataclass
from typing import List

import numpy as np

from scipy.interpolate import interp1d

from robosim.simulator.robot_simulator import (
    Robot,
    ConfigurationSpaceType,
    WorkSpaceType,
)

from robosim.robo_state import JointState, Pose


def batched_2d_to_3d(xs: np.ndarray, z_value=0.0):
    # xs.shape == B x ... x 2
    return np.concatenate([xs, z_value * np.ones((*xs.shape[:-1], 1))], axis=-1)


@dataclass
class Trajectory:
    states: List[JointState]

    def get(self, joint_state_names: List[str]) -> List[List[float]]:
        return [state.get(joint_state_names) for state in self.states]


def interpolate_trajectory(start, target):
    fst = np.array(start)
    snd = np.array(target)
    linfit = interp1d([0, 1], np.vstack([fst, snd]), axis=0)
    return linfit


class EndEffectorTrajectory:
    def __init__(self, xs: WorkSpaceType):
        # must be in the form of [B x 2] or [B x 3]
        self.xs = np.array(xs)
        if self.xs.shape[-1] not in (2, 3):
            raise ValueError()

    def to_qs(self, robot: Robot, z_value: float = 0.2) -> ConfigurationSpaceType:
        xs = self.xs
        if xs.shape[-1] == 2:
            xs = batched_2d_to_3d(xs, z_value=z_value)
        return robot.ee_xs_to_qs(xs)

    def to_configuration_space(self, robot: Robot, **kwargs) -> Trajectory:
        qs = self.to_qs(robot=robot, **kwargs)
        return Trajectory(
            states=[JointState(name=robot.target_joint_names, position=q) for q in qs]
        )

    def __repr__(self):
        return f"{self.xs}"
