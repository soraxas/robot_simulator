import math

from typing import List

import quaternion
import numpy as np


class Transform:
    def __init__(self, mat=None, quat: quaternion.quaternion = None, pos=None):
        if mat is None:
            self.matrix = np.identity(4)
            if quat is not None and pos is not None:
                self.matrix[:3, :3] = quaternion.as_rotation_matrix(quat)
                self.matrix[:3, 3] = pos
        else:
            self.matrix = np.copy(mat)

    def __str__(self):
        return self.matrix.__str__()

    def __repr__(self):
        return self.matrix.__repr__()

    def quat_2_mat(self, quat, pos):
        """Conversion quaternion vers matrix."""
        self.matrix[:3, :3] = quaternion.as_rotation_matrix(quat)
        self.matrix[:3, 3] = pos

    def inverse(self):
        return Transform(np.linalg.inv(self.matrix))

    def __invert__(self):
        return Transform(np.linalg.inv(self.matrix))

    def __sub__(self, other):
        return self.composition(~other)

    def __isub__(self, other):
        return self.composition(~other)

    def quaternion(self) -> quaternion.quaternion:
        return quaternion.from_rotation_matrix(self.matrix)

    def position(self) -> List[float]:
        return self.matrix[:3, 3]

    def composition(self, tr):
        return Transform(mat=np.dot(self.matrix, tr.matrix))

    def __mul__(self, other):
        return self.composition(other)

    def __imul__(self, other):
        self.matrix = self.matrix.dot(other.matrix)
        return self

    def relative_transform(self, other) -> "Transform":
        return ~other.composition(self)

    def projection(self, pt):
        if len(pt) == 3:
            return self.matrix.dot(pt + [1])
        else:
            return self.matrix.dot(pt)


def rotation_matrix(axe, angle):
    matrix = np.identity(4)
    if axe == "x":
        matrix[1, 1] = math.cos(angle)
        matrix[1, 2] = -math.sin(angle)
        matrix[2, 1] = math.sin(angle)
        matrix[2, 2] = math.cos(angle)
    elif axe == "y":
        matrix[0, 0] = math.cos(angle)
        matrix[0, 2] = math.sin(angle)
        matrix[2, 0] = -math.sin(angle)
        matrix[2, 2] = math.cos(angle)
    elif axe == "z":
        matrix[0, 0] = math.cos(angle)
        matrix[0, 1] = -math.sin(angle)
        matrix[1, 0] = math.sin(angle)
        matrix[1, 1] = math.cos(angle)
    return matrix


def translation_matrix(tr):
    matrix = np.identity(4)
    matrix[:3, 3] = np.asarray(tr)
    return matrix


#
# @dataclass
# class Quaternion:
#     values: List[float]
#
#     def __len__(self):
#         return len(self.values)
#
#     def __mul__(self, other: "Quaternion") -> "Quaternion":
#         x1, y1, z1, w1 = self.values
#         x2, y2, z2, w2 = other.values
#
#         w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
#         x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
#         y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
#         z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
#         return Quaternion([x, y, z, w])
#
#     def conjugate(self):
#         x, y, z, w = self.values
#         return Quaternion([-x, -y, -z, w])
#
#     def rotational_matrix(self):
#         qx, qy, qz, qw = self.values
#
#         m00 = 1 - 2 * qy ** 2 - 2 * qz ** 2
#         m01 = 2 * qx * qy - 2 * qz * qw
#         m02 = 2 * qx * qz + 2 * qy * qw
#         m10 = 2 * qx * qy + 2 * qz * qw
#         m11 = 1 - 2 * qx ** 2 - 2 * qz ** 2
#         m12 = 2 * qy * qz - 2 * qx * qw
#         m20 = 2 * qx * qz - 2 * qy * qw
#         m21 = 2 * qy * qz + 2 * qx * qw
#         m22 = 1 - 2 * qx ** 2 - 2 * qy ** 2
#         result = [[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]]
#
#         return result
