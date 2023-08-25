from typing import Iterable
from pathlib import Path
from abc import ABC
import numpy as np

from robosim.robo_trajectory import EndEffectorTrajectory


class DemonstratedWorldSpaceTrajectories:
    def __init__(self, demonstrations: np.ndarray, name: str):
        # demonstrations must be a numpy array with ..x..x..x[2/3]

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


class WorldSpaceTrajectoryDataset(ABC):
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)

    def __iter__(self) -> Iterable[DemonstratedWorldSpaceTrajectories]:
        raise NotImplementedError()
