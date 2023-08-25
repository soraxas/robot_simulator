from pathlib import Path
import numpy as np
from robosim.utils import get_robot_resources_root

from robosim.trajectory_demonstration import (
    WorldSpaceTrajectoryDataset,
    DemonstratedWorldSpaceTrajectories,
)

_list_of_files = [
    "h.npy",
    "e.npy",
    "l.npy",
    "l.npy",
    "o.npy",
]


class ClfdHelloworldTrajectoryDataset(WorldSpaceTrajectoryDataset):
    def __init__(self, root_path: str = None):
        if root_path is None:
            root_path = (
                get_robot_resources_root()
                / "dataset_traj-demonstrations/clfd/robot_hello_world/processed_demos"
            )
        self.root_path = Path(root_path)

    def __iter__(self):
        for fname in _list_of_files:
            # original dataset in cm
            xs = np.load(self.root_path / fname) / 100
            yield DemonstratedWorldSpaceTrajectories(
                demonstrations=xs,
                name=fname,
            )
