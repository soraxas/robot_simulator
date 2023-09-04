from abc import ABC, abstractmethod
from typing import List

import pybullet_tools.utils as pu

from robosim.robo_state import Pose


class PybulletSceneObject(ABC):
    pose: Pose
    handle = None

    def __init__(self, pose: Pose):
        self.pose = pose

    @abstractmethod
    def _build(self) -> int:
        pass

    def build(self):
        self.handle = self._build()


class BoxPrimitive(PybulletSceneObject):
    def __init__(self, dim: List[float], pose: Pose):
        super().__init__(pose=pose)
        assert len(dim) == 3
        self.dim = dim

    def _build(self):
        return pu.create_box(*self.dim, pose=self.pose)


class CylinderPrimitive(PybulletSceneObject):
    def __init__(self, radius: float, height: float, pose: Pose):
        super().__init__(pose=pose)
        self.radius = radius
        self.height = height

    def _build(self):
        return pu.create_cylinder(
            radius=self.radius, height=self.height, pose=self.pose
        )


class Mesh(PybulletSceneObject):
    def __init__(self, mesh, pose: Pose):
        super().__init__(pose=pose)

        # mesh should be a tuple of (vertices, triangles)
        self.mesh = mesh

    def _build(self):
        return pu.create_mesh(self.mesh, pose=self.pose)
