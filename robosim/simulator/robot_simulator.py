from functools import cached_property
from typing import Optional, List, Tuple, Union, Iterable

import pybullet as p
import pybullet_data
import pybullet_tools.utils as pu
import torch
import numpy as np

from robosim.utils import get_robot_resources_root
from robosim.controller_output import ControllerJointType, AbstractController
from robosim.controller_output.pybullet_controller import PyBulletController

from .data_types import WorkSpaceType, ConfigurationSpaceType


ROBOT_RESOURCE_ROOT = get_robot_resources_root() / "robot_model"


class Robot:
    def __init__(
        self,
        urdf_path: str,
        target_link_names: List[str],
        target_joint_names: List[str],
        end_effector_link_name: str,
        start_position: List[float] = (0, 0, 0),
        start_orientation: List[float] = (0, 0, 0, 1),
        default_qs: List[float] = None,
        device=None,
        include_plane=True,
        p_client=p.DIRECT,
        has_gravity=False,
        setup_learnable_model: bool = False,
        controller: Optional[AbstractController] = None,
    ):
        self.urdf_path = urdf_path
        self.target_link_names = target_link_names
        self.target_joint_names = target_joint_names

        assert len(start_position) == 3
        assert len(start_orientation) == 4

        self.learnable_robot_model = None
        if setup_learnable_model:
            from . import learnable_robot

            self.learnable_robot_model = learnable_robot.setup_learnable_robot(
                urdf_path=urdf_path,
                start_position=start_position,
                rotational_matrix=p.getMatrixFromQuaternion(start_orientation),
                device=device,
            )

        ###############################################################
        # setup pybullet
        self.physicsClient = p.connect(p_client)
        # physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version

        if has_gravity:
            p.setGravity(0, 0, -9.81)
            # p.setGravity(0,0,-10)
        else:
            pu.disable_gravity()

        if controller is None:
            controller = PyBulletController()
        self.controller: AbstractController = controller
        self.controller.set_robot(self)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        if include_plane:
            planeId = p.loadURDF("plane.urdf")
        # startOrientation = p.getQuaternionFromEualer([0, 0, 0])

        self.pyb_robot_id = p.loadURDF(
            urdf_path, start_position, start_orientation, useFixedBase=True
        )
        self.ee_joint_idx = pu.link_from_name(self.pyb_robot_id, end_effector_link_name)
        ###############################################################

        if not all(
            name in self.joints_names_to_index_mapping for name in target_joint_names
        ):
            raise ValueError(
                f"Not all target joint exists for the robot. "
                f"Given {target_joint_names}. "
                f"Contains {self.joints_names}."
            )

        if default_qs is not None:
            self.set_qs(default_qs)

    @property
    def dof(self):
        return len(self.target_joint_names)

    def joint_name_to_indexes(self, joint_name: List[str]):
        return [
            pu.get_joint(self.pyb_robot_id, joint_name) for joint_name in joint_name
        ]

    def set_qs(
        self,
        qs: ConfigurationSpaceType,
        joint_indexes: List[int] = None,
        joint_names: List[str] = None,
    ):
        self.controller.set_qs(
            qs=qs, joint_indexes=joint_indexes, joint_names=joint_names
        )

    def ee_xs_to_all_qs(
        self, xs: WorkSpaceType, reference_orientation: Optional[List[float]] = None
    ) -> ConfigurationSpaceType:
        """
        Given xs with size [b x 3], returns the corresponding configurations
        """
        assert xs.shape[-1] == 3

        return [
            pu.inverse_kinematics_helper(
                self.pyb_robot_id, self.ee_joint_idx, (x, reference_orientation)
            )
            for x in xs
        ]

    def ee_xs_to_qs(
        self, xs: WorkSpaceType, reference_orientation: Optional[List[float]] = None
    ) -> ConfigurationSpaceType:
        """
        This is a wrapper that only returns the requested target qs.
        """

        def filter_unneeded_qs(all_qs):
            mapped_qs = []
            for _qs in all_qs:
                mapped_qs.append([])
                for idx in self.target_joint_indices_pybullet:
                    mapped_qs[-1].append(_qs[idx])
            return mapped_qs

        # handle batched version.
        if hasattr(xs, "shape") and len(xs.shape) == 3:
            batch_size = xs.shape[0]
            return torch.Tensor(
                [
                    filter_unneeded_qs(
                        self.ee_xs_to_all_qs(
                            xs[i, ...], reference_orientation=reference_orientation
                        )
                    )
                    for i in range(batch_size)
                ]
            )
        # else
        return filter_unneeded_qs(self.ee_xs_to_all_qs(xs, reference_orientation))

    def qs_to_joints_xs(self, qs: ConfigurationSpaceType) -> WorkSpaceType:
        """
        Given batch of qs [b1 x ... x bn x d]
        Returns a batch of pos  [b1 x ... x bn x j x 3], where J is the number of joint from root to ee
        e.g.,
            input can be [20 x 6] for a 6dof robot,
            output can be [20 x 8 x 3], where 8 is the number of join in-between, and 3 is the xyz dimension.
        """
        # ignore orientation
        # handle batch.
        batch_shape = qs.shape[:-1]
        dimensionality = qs.shape[-1]
        qs = qs.reshape(-1, dimensionality)

        result = torch.stack([pose[0] for pose in self.qs_to_joints_pose(qs)])
        # shape of j, b, 3, where j is the number of joint from root to ee
        result = torch.swapaxes(result, 0, 1)  # shape is now b, j, 3
        return result.reshape(*batch_shape, *result.shape[-2:])

    def qs_to_joints_pose(
        self, qs: ConfigurationSpaceType
    ) -> List[Tuple[torch.Tensor]]:
        """
        Given batch of qs
        Returns a list of pose (i.e., pos in R^3 and orientation in R^4)
        """
        # defaulting the non-used index as zero
        # if len(self.joints_names) == qs.shape[-1]:
        #     mapped_qs = qs
        # else:

        mapped_qs = torch.zeros(
            (*qs.shape[:-1], self.learnable_robot_model._n_dofs),
            device=qs.device,
            dtype=qs.dtype,
        )
        for i, idx in enumerate(self.target_joint_indices):
            mapped_qs[..., idx] = qs[..., i]

        joints_xs = self.learnable_robot_model.compute_forward_kinematics_all_links(
            mapped_qs
        )
        # base had been skipped
        return [joints_xs[name] for name in self.target_link_names]

    def print_info(self):
        print(pu.get_body_info(self.pyb_robot_id))

        for j_idx in range(pu.get_num_joints(self.pyb_robot_id)):
            print(f"===== Joint {j_idx} =====")
            print(pu.get_joint_info(self.pyb_robot_id, j_idx))

    def get_all_joints_limits(self, as_tensor=True):
        """Returns all available joints limits"""
        lowers = []
        uppers = []
        for limits in self.learnable_robot_model.get_joint_limits():
            lowers.append(limits["lower"])
            uppers.append(limits["upper"])
        if as_tensor:
            lowers, uppers = torch.Tensor(lowers), torch.Tensor(uppers)
        return lowers, uppers

    @cached_property
    def joints_names(self):
        names = []
        for i in range(pu.get_num_joints(self.pyb_robot_id)):
            if not pu.is_fixed(self.pyb_robot_id, i):
                names.append(pu.get_joint_name(self.pyb_robot_id, i))
        return names

    @cached_property
    def joints_names_to_index_mapping(self):
        return {name: i for i, name in enumerate(self.joints_names)}

    # TODO: FIXME: :(
    @cached_property
    def joints_names_learnable(self):
        return self.learnable_robot_model.get_joint_names()

    @cached_property
    def joints_names_to_index_mapping_learnable(self):
        return {name: i for i, name in enumerate(self.joints_names_learnable)}

    @cached_property
    def target_joint_indices(self):
        target_joint_indices = []
        for name in self.target_joint_names:
            for i, joint_name in enumerate(
                self.learnable_robot_model.get_joint_names()
            ):
                if joint_name == name:
                    target_joint_indices.append(i)
                    break
            else:
                print(f">> Target joint with name {name} not found!")
                exit(1)

        return target_joint_indices

        # return [
        #     self.joints_names_to_index_mapping[name] for name in self.target_joint_names
        # ]

    @cached_property
    def target_joint_indices_pybullet(self):
        return [
            self.joints_names_to_index_mapping[name] for name in self.target_joint_names
        ]

    def get_joints_limits(self, as_tensor=True):
        """Returns all requested (target) joints limits"""
        all_joint_limits = self.get_all_joints_limits(as_tensor=False)

        lowers = []
        uppers = []
        for target_joint_name in self.target_joint_names:
            idx = self.joints_names.index(target_joint_name)
            lowers.append(all_joint_limits[0][idx])
            uppers.append(all_joint_limits[1][idx])
        if as_tensor:
            lowers, uppers = torch.Tensor(lowers), torch.Tensor(uppers)
        return lowers, uppers

    def get_collision_functor(
        self,
        obstacles: Optional[List[int]] = None,
        attachments=None,
        self_collisions: bool = True,
        disabled_collisions=None,
        custom_limits=None,
        use_aabb=False,
        cache=False,
        max_distance=pu.MAX_DISTANCE,
        check_joint_limits=True,
        **kwargs,
    ):
        if custom_limits is None:
            custom_limits = dict()
        if disabled_collisions is None:
            disabled_collisions = set()
        if attachments is None:
            attachments = list()
        if obstacles is None:
            obstacles = list()

        joint_indexes = self.joint_name_to_indexes(self.target_joint_names)
        if not self_collisions:
            check_link_pairs = []
        else:
            check_link_pairs = pu.get_self_link_pairs(
                self.pyb_robot_id, joint_indexes, disabled_collisions
            )

        moving_links = frozenset(
            link
            for link in pu.get_moving_links(self.pyb_robot_id, joint_indexes)
            if pu.can_collide(self.pyb_robot_id, link)
        )  # TODO: propagate elsewhere
        attached_bodies = [attachment.child for attachment in attachments]
        moving_bodies = [pu.CollisionPair(self.pyb_robot_id, moving_links)] + list(
            map(pu.parse_body, attached_bodies)
        )
        get_obstacle_aabb = pu.cached_fn(
            pu.get_buffered_aabb, cache=cache, max_distance=max_distance / 2.0, **kwargs
        )
        if check_joint_limits:
            limits_fn = pu.get_limits_fn(
                self.pyb_robot_id, joint_indexes, custom_limits=custom_limits
            )
        else:
            limits_fn = lambda *args: False

        def collision_fn(q, verbose=False):
            if limits_fn(q):
                return True
            pu.set_joint_positions(self.pyb_robot_id, joint_indexes, q)
            for attachment in attachments:
                attachment.assign()
            # wait_for_duration(1e-2)
            get_moving_aabb = pu.cached_fn(
                pu.get_buffered_aabb,
                cache=True,
                max_distance=max_distance / 2.0,
                **kwargs,
            )

            for link1, link2 in check_link_pairs:
                # Self-collisions should not have the max_distance parameter
                if (
                    not use_aabb
                    or pu.aabb_overlap(
                        get_moving_aabb(self.pyb_robot_id),
                        get_moving_aabb(self.pyb_robot_id),
                    )
                ) and pu.pairwise_link_collision(
                    self.pyb_robot_id, link1, self.pyb_robot_id, link2
                ):  # , **kwargs):
                    return True

            for body1, body2 in pu.product(moving_bodies, obstacles):
                if (
                    not use_aabb
                    or pu.aabb_overlap(get_moving_aabb(body1), get_obstacle_aabb(body2))
                ) and pu.pairwise_collision(body1, body2, **kwargs):
                    return True
            return False

        return collision_fn

    def destroy(self):
        p.disconnect(self.physicsClient)

    # def get_joints_limits(self):
    #     lowers = []
    #     uppers = []
    #     for j_idx in range(pu.get_num_joints(self.pyb_robot_id)):
    #         limit = pu.get_joint_limits(self.pyb_robot_id, j_idx)
    #         lowers.append(limit[0])
    #         uppers.append(limit[1])
    #     return torch.Tensor(lowers),torch.Tensor(uppers)


class PandaRobot(Robot):
    def __init__(self, **kwargs):
        self.urdf_path = ROBOT_RESOURCE_ROOT / "panda/urdf/panda.urdf"
        # choose links to operate
        target_link_names = [
            # "panda_link0",
            "panda_link1",
            "panda_link2",
            "panda_link3",
            "panda_link4",
            "panda_link5",
            "panda_link6",
            "panda_link7",
            "panda_link8",
            "panda_hand",
        ]
        target_joint_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
            # "panda_joint8",
            # "panda_hand_joint",
        ]
        super().__init__(
            urdf_path=str(self.urdf_path),
            target_link_names=target_link_names,
            target_joint_names=target_joint_names,
            end_effector_link_name="panda_hand",
            **kwargs,
        )


class KinovaRobot(Robot):
    def __init__(self, **kwargs):
        self.urdf_path = ROBOT_RESOURCE_ROOT / "kinova/urdf/jaco_clean.urdf"
        # choose links to operate
        if "target_link_names" not in kwargs:
            kwargs["target_link_names"] = [
                # "j2n6s300_link_base",
                "j2n6s300_link_1",
                "j2n6s300_link_2",
                "j2n6s300_link_3",
                "j2n6s300_link_4",
                "j2n6s300_link_5",
                "j2n6s300_link_6",
                "j2n6s300_end_effector",
                "j2n6s300_link_ee",
            ]
        if "target_joint_names" not in kwargs:
            kwargs["target_joint_names"] = [
                "j2n6s300_joint_1",
                "j2n6s300_joint_2",
                "j2n6s300_joint_3",
                "j2n6s300_joint_4",
                "j2n6s300_joint_5",
                "j2n6s300_joint_6",
                # "j2n6s300_joint_end_effector",
                "j2n6s300_joint_finger_1",
                # "j2n6s300_joint_finger_tip_1",
                "j2n6s300_joint_finger_2",
                # "j2n6s300_joint_finger_tip_2",
                "j2n6s300_joint_finger_3",
                # "j2n6s300_joint_finger_tip_3",
                # "panda_joint8",
                # "panda_hand_joint",
            ]
            if "default_qs" not in kwargs:
                kwargs["default_qs"] = [0, np.pi, np.pi, 0, 0, 0, 0, 0, 0]
                # kwargs["default_qs"] = [0, np.pi, np.pi, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        super().__init__(
            urdf_path=str(self.urdf_path),
            end_effector_link_name="j2n6s300_link_ee",
            **kwargs,
        )
