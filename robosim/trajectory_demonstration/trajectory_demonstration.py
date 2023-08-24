from pathlib import Path
import numpy as np

from robosim.robo_trajectory import EndEffectorTrajectory

list_of_files = [
    "h.npy",
    "e.npy",
    "l.npy",
    "l.npy",
    "o.npy",
]


class DemonstratedWorldSpaceTrajectories:
    def __init__(self, demonstrations: np.ndarray, name: str):
        self.name = name
        assert demonstrations.shape[-1] in (2, 3)
        assert len(demonstrations.shape) >= 2

        if len(demonstrations.shape) == 2:
            # unsqueeze
            demonstrations = demonstrations.reshape(1, *demonstrations.shape)
        elif len(demonstrations.shape) > 3:
            demonstrations = demonstrations.reshape(
                -1, demonstrations.shape[-2], demonstrations.shape[-1]
            )
        self.demonstrations = demonstrations

    def __len__(self):
        return self.demonstrations.shape[0]

    def __iter__(self):
        demonstrations = self.demonstrations

        for i in range(demonstrations.shape[0]):
            yield EndEffectorTrajectory(demonstrations[i, :, :])


class WorldSpaceTrajectoryDataset:
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)

    def __iter__(self):
        for fname in list_of_files:
            # original dataset in cm
            xs = np.load(self.root_path / fname) / 100
            yield DemonstratedWorldSpaceTrajectories(
                demonstrations=xs,
                name=fname,
            )
