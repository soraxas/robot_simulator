from typing import List

import numpy as np
import torch
from differentiable_robot_model import DifferentiableRobotModel


def setup_learnable_robot(
    urdf_path: str,
    start_position: List[float] = (0, 0, 0),
    rotational_matrix: np.ndarray = None,
    device=None,
):
    ###############################################################
    # setup differentiable robot
    learnable_robot_model = DifferentiableRobotModel(
        urdf_path, name="my_robot", device=device
    ).eval()
    # setup pos and rot
    learnable_robot_model._bodies[0].pose._trans[:, ...] = torch.Tensor(start_position)
    learnable_robot_model._bodies[0].pose._rot[:, ...] = torch.Tensor(
        rotational_matrix
    ).reshape(3, 3)
    return learnable_robot_model
