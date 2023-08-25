import os
import random
import time
from pathlib import Path

import numpy as np
import torch


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
