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

from robosim.robo_trajectory import Trajectory, interpolate_trajectory
from robosim.robot_scene.objects import PybulletSceneObject
from robosim.utils import get_project_root

from robosim.simulator.robot_simulator import Robot, ConfigurationSpaceType

from robosim.robo_state import JointState, Pose


@dataclass
class PathRequest:
    start_state: JointState
    target_state: JointState


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

    def add_object(self, scene_object: PybulletSceneObject):
        self.added_bodies.append(scene_object.build())

    def play(
        self,
        trajectory: Trajectory,
        target_joint_names: List[str] = None,
        interpolate_step: int = 50,
        delay_between_interpolated_joint: float = 0.02,
        delay_between_joint: float = 2.0,
        callback: Callable[[ConfigurationSpaceType, int], None] = None,
    ):
        # if target_joint_names is not given, it will default to the joint names given
        # in the first joint state of the trajectory
        if target_joint_names is None:
            target_joint_names = list(trajectory.states[0].name)

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
