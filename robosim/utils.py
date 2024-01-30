import os
import random
import time
from pathlib import Path

import numpy as np
import torch

color_qualitative = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]


def generator(iterable):
    while True:
        yield from iterable


def generate_seeds(n):
    return [random.randint(0, 2**32 - 1) for _ in range(n)]


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_project_root() -> Path:
    return Path(__file__).parent.parent.resolve()


def get_robot_resources_root() -> Path:
    return get_project_root() / "robot_resources"


def get_robot_resources_root() -> Path:
    return get_project_root() / "robot_resources"

    # ctx_mgr = PybulletDebugCtxMgr(plot_robot_ee_in_pybullet(robot=robot, qs=qs))

    # import numpy as np

    # for demonstrations in demonstration_dataset:
    #     _ids = []
    #     color_gen = generator(color_qualitative)


ID = 0


def plot_robot_ee_in_pybullet(
    robot,
    ee_xs=None,
    qs=None,
    traj_color=[0.38823529, 0.43137255, 0.98039216],
    point_size=5.0,
    different_color_per_batch: bool = True,
):
    global ID
    from pybullet_tools.utils import NULL_ID, CLIENT, get_lifetime, RGB
    import torch
    import pybullet as p

    ee_xs = ee_xs.reshape(-1, 3)
    return p.addUserDebugPoints(
        ee_xs,
        pointColorsRGB=[traj_color] * ee_xs.shape[0],
        pointSize=point_size,
        lifeTime=get_lifetime(None),
        parentObjectUniqueId=NULL_ID,
        parentLinkIndex=NULL_ID,
        physicsClientId=CLIENT,
    )


def hex_to_rgb(hex_string, zero_to_1: bool = True):
    hex_string = hex_string.lstrip("#")
    rgb = np.array([int(hex_string[i : i + 2], 16) for i in (0, 2, 4)])
    if zero_to_1:
        rgb = rgb / 255.0
    return rgb


class PybulletDebugCtxMgr:
    def __init__(self, *debug_ids: int):
        self.debug_ids = debug_ids

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        import pybullet_tools.utils as pu

        for id in self.debug_ids:
            pu.remove_debug(id)


class EmptyCtxMgr:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
