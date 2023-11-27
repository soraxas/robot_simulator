import glob
import time
from dataclasses import dataclass
from functools import cached_property
from typing import List, Callable, Union

import numpy as np

import pybullet as p
import pybullet_tools.utils as pu
import yaml
from scipy.interpolate import interp1d

from robosim.robo_trajectory import Trajectory, interpolate_trajectory
from robosim.utils import get_project_root

from robosim.simulator.robot_simulator import Robot, ConfigurationSpaceType
from robosim.robo_state import JointState, Pose

from .objects import PybulletSceneObject


@dataclass
class PathRequest:
    start_state: JointState
    target_state: JointState


USE_PYBULLET_CONTROL = True


# USE_PYBULLET_CONTROL = False


class Parameter:
    def __init__(
        self,
        val: float,
        name: str = "",
        lower: float = 0,
        upper: float = 1,
        type_caster: Callable = float,
    ):
        self.val = val
        if USE_PYBULLET_CONTROL:
            EPSILON = 1e-5
            EPSILON = 0
            self.handle = None
            self.name = name
            # lower and upper bound MUST bound the val
            self.lower = min(lower, self.val - EPSILON)
            self.upper = max(upper, self.val + EPSILON)
            self.type_caster = type_caster

    def get(self) -> Union[int, float]:
        if USE_PYBULLET_CONTROL:
            if self.handle is None:
                self.handle = pu.add_parameter(
                    self.name, float(self.lower), float(self.upper), float(self.val)
                )
            # return 5
            val = pu.read_parameter(self.handle)
            return self.type_caster(val)
        else:
            return self.val

    @classmethod
    def ensure(cls, val: Union[float, int, "Parameter"]):
        if isinstance(val, Parameter):
            return val
        return Parameter(val=val)


PARAMS = dict()


def GetParameter(name: str, **kwargs):
    if name not in PARAMS:
        PARAMS[name] = Parameter(name=name, **kwargs)
    return PARAMS[name]


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
        interpolate_step = GetParameter(
            name="interpolate_step",
            val=interpolate_step,
            lower=0,
            upper=50,
            type_caster=int,
        )
        delay_between_interpolated_joint = GetParameter(
            name="delay_between_interpolated_joint",
            val=delay_between_interpolated_joint,
            lower=0,
            upper=0.5,
        )
        delay_between_joint = GetParameter(
            name="delay_between_joint",
            val=delay_between_joint,
            lower=0,
            upper=0.01,
        )

        # if target_joint_names is not given, it will default to the joint names given
        # in the first joint state of the trajectory
        if target_joint_names is None:
            target_joint_names = list(trajectory.states[0].name)

        target_joint_indexes = self.robot.joint_name_to_indexes(target_joint_names)

        last_qs = None
        for i, qs in enumerate(trajectory.get(target_joint_names)):
            if last_qs is not None:
                interp = interpolate_trajectory(last_qs, qs)
                ts = np.linspace(0, 1, num=interpolate_step.get())
                for t in ts:
                    self.robot.set_qs(interp(t), target_joint_indexes)
                    time.sleep(delay_between_interpolated_joint.get())

            last_qs = qs
            self.robot.set_qs(qs, target_joint_indexes)
            if callback:
                callback(np.array(qs), i)
            time.sleep(delay_between_joint.get())
            time.sleep(delay_between_joint.get())
            time.sleep(delay_between_joint.get())
