import glob
import time
from dataclasses import dataclass
from functools import cached_property
from typing import List, Callable

import numpy as np

import pybullet as p
import pybullet_tools.utils as pu
import yaml
from scipy.interpolate import interp1d

from robosim.utils import get_project_root

from robosim.simulator.robot_simulator import Robot, ConfigurationSpaceType

from robosim.robo_state import JointState, Pose


@dataclass
class PathRequest:
    start_state: JointState
    target_state: JointState


@dataclass
class Trajectory:
    states: List[JointState]

    def get(self, joint_state_names: List[str]):
        return [state.get(joint_state_names) for state in self.states]


def interpolate_trajectory(start, target):
    fst = np.array(start)
    snd = np.array(target)
    linfit = interp1d([0, 1], np.vstack([fst, snd]), axis=0)
    return linfit


class RobotScene:
    def __init__(self, robot: Robot):
        self.robot = robot
        self.added_bodies = []

    @cached_property
    def robot_base_offset(self) -> Pose:
        return Pose(
            [0, 0, 0],
            [0, 0, 0, 1],
        )

    def clear(self):
        while len(self.added_bodies) > 0:
            bid = self.added_bodies.pop(0)
            p.removeBody(bid)

    def build_scene(self):
        return list(self.added_bodies)

    def play(
        self,
        trajectory: Trajectory,
        target_joint_names: List[str],
        interpolate_step: int = 50,
        delay_between_interpolated_joint: float = 0.02,
        delay_between_joint: float = 2.0,
        callback: Callable[[ConfigurationSpaceType, int], None] = None,
    ):
        target_joint_indexes = self.robot.joint_name_to_indexes(target_joint_names)

        last_qs = None
        for i, qs in enumerate(trajectory.get(target_joint_names)):
            if last_qs is not None:
                interp = interpolate_trajectory(last_qs, qs)
                ts = np.linspace(0, 1, num=interpolate_step)
                for t in ts:
                    self.robot.set_qs(interp(t), target_joint_indexes)
                    time.sleep(delay_between_interpolated_joint)

            last_qs = qs
            self.robot.set_qs(qs, target_joint_indexes)
            if callback:
                callback(np.array(qs), i)
            time.sleep(delay_between_joint)
