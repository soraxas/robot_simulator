import glob
from dataclasses import dataclass
from functools import cached_property
from os import path
from pathlib import Path

import pybullet as p
import pybullet_tools.utils as pu
import yaml

from robosim.utils import get_project_root

from robosim.simulator.robot_simulator import Robot
from . import Trajectory

from .robot_scene import RobotScene, PathRequest
from robosim.robo_state import JointState, Pose

this_directory = Path(path.abspath(path.dirname(__file__)))

tag_names = [
    "bookshelf_small_panda",
    "bookshelf_tall_panda",
    "bookshelf_thin_panda",
    "box_panda",
    "cage_panda",
    "kitchen_panda",
    "table_bars_panda",
    "table_pick_panda",
    "table_under_pick_panda",
]


def path_request_from_yaml(fname) -> "PathRequest":
    with open(fname, "r") as f:
        obj = yaml.safe_load(f)
    return PathRequest(
        JointState(
            obj["start_state"]["joint_state"]["name"],
            obj["start_state"]["joint_state"]["position"],
        ),
        JointState(
            [j["joint_name"] for j in obj["goal_constraints"][0]["joint_constraints"]],
            [j["position"] for j in obj["goal_constraints"][0]["joint_constraints"]],
        ),
    )


def trajectory_from_yaml(fname: str) -> "Trajectory":
    with open(fname, "r") as f:
        obj = yaml.safe_load(f)
    return Trajectory(
        [
            JointState(obj["joint_trajectory"]["joint_names"], point["positions"])
            for point in obj["joint_trajectory"]["points"]
        ]
    )


root_path = Path(
    "/home/tin/research/stein_mpc/robot_simulator/clfd/datasets/robot_hello_world/processed_demos"
)

import numpy as np

demonstrations = np.load(root_path / "h.npy")

print(demonstrations)
print(demonstrations.shape)
exit()


class CLFDScene(RobotScene):
    def __init__(self, robot: Robot, tag_name: str):
        super().__init__(robot=robot)
        self.tag_name = tag_name

    @cached_property
    def config_path(self):
        return get_project_root() / "robodata" / f"{self.tag_name}-config.yaml"

    @cached_property
    def robot_base_offset(self) -> Pose:
        with open(self.config_path, "r") as f:
            yamlobj = yaml.safe_load(f)
        return Pose(
            yamlobj["base_offset"]["position"],
            yamlobj["base_offset"]["orientation"],
        )

    @cached_property
    def scene_path(self):
        return get_project_root() / "robodata" / f"{self.tag_name}-scene0001.yaml"

    @cached_property
    def weight_path(self):
        return (
            get_project_root()
            / "robodata"
            / f"{self.tag_name}-scene0001_continuous-occmap-weight.ckpt"
        )

    @cached_property
    def dataset_path(self):
        return (
            get_project_root() / "robodata" / f"{self.tag_name}-scene0001_dataset.csv"
        )

    def __len__(self):
        return len(self.trajectory_paths)

    @cached_property
    def trajectory_paths(self):
        return sorted(
            glob.glob(
                str(
                    get_project_root()
                    / "robodata"
                    / f"{self.tag_name}-scene0001_path*.yaml"
                )
            )
        )

    @cached_property
    def request_paths(self):
        return sorted(
            glob.glob(
                str(
                    get_project_root()
                    / "robodata"
                    / f"{self.tag_name}-scene0001_request*.yaml"
                )
            )
        )

    def build_scene(self):
        with open(self.scene_path, "r") as stream:
            yamlobj = yaml.safe_load(stream)

        if "fixed_frame_transforms" in yamlobj:
            _base_transform = yamlobj["fixed_frame_transforms"][0]["transform"]
            p.resetBasePositionAndOrientation(
                self.robot.pyb_robot_id,
                _base_transform["translation"],
                _base_transform["rotation"],
            )

        for obj in yamlobj["world"]["collision_objects"]:
            if "primitives" in obj:
                assert len(obj["primitives"]) == 1

                # primitive objects
                _type = obj["primitives"][0]["type"]
                _ = obj["primitive_poses"][0]

                # base frame
                pose = Pose(
                    obj["pose"]["position"],
                    obj["pose"]["orientation"]
                    # Quaternion([0,0,0,1])
                )
                # transform from the base frame
                pose = pose.composite(Pose(_["position"], _["orientation"]))

                dim = obj["primitives"][0]["dimensions"]
                if _type == "box":
                    self.added_bodies.append(pu.create_box(*dim, pose=pose))
                elif _type == "cylinder":
                    dim = obj["primitives"][0]["dimensions"]
                    self.added_bodies.append(
                        pu.create_cylinder(radius=dim[1], height=dim[0], pose=pose)
                    )
                else:
                    raise NotImplementedError(_type)

            elif "meshes" in obj:
                assert len(obj["meshes"]) == 1
                # meshes
                _ = obj["mesh_poses"][0]

                # base frame
                pose = Pose(
                    obj["pose"]["position"],
                    obj["pose"]["orientation"]
                    # Quaternion([0,0,0,1])
                )

                # transform from the base frame
                pose = pose.composite(Pose(_["position"], _["orientation"]))

                mesh = obj["meshes"][0]["vertices"], obj["meshes"][0]["triangles"]
                self.added_bodies.append(pu.create_mesh(mesh, pose=pose))

            else:
                raise NotImplementedError(str(obj))
        return list(self.added_bodies)
