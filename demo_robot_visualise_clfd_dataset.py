import argparse
import time
from pathlib import Path

import pybullet as p
import pybullet_tools.utils as pu

import time
from robosim.scene import JointState, robot_scene
from robosim.simulator.robot_simulator import PandaRobot, KinovaRobot
from robosim.robo_trajectory import Trajectory, EndEffectorTrajectory
from robosim.trajectory_demonstration.clfd_helloworld_demonstration import (
    ClfdHelloworldTrajectoryDataset,
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "-d0", "--delay-between-interpolated-joint", default=0.01, type=float
)
parser.add_argument("-d1", "--delay-between-joint", default=0.00025, type=float)
parser.add_argument("-d2", "--delay-between-solution", default=0.15, type=float)
parser.add_argument("-d3", "--delay-between-scene", default=2, type=float)
parser.add_argument("-s", "--interpolate-step", default=0, type=int)
parser.add_argument("-n", "--num-solution-to-vis", default=100, type=int)
parser.add_argument(
    "-r",
    "--visualise-request",
    help="visualise the request start and target directly, "
    "instead of the example trajectory",
    action="store_true",
)

if __name__ == "__main__":
    args = parser.parse_args()

    print("\n\n")

    robot = KinovaRobot(
        device="cpu",
        # p_client=p.DIRECT,
        p_client=p.GUI,
        include_plane=False,
        setup_learnable_model=True,
    )
    demonstration_dataset = ClfdHelloworldTrajectoryDataset()

    # close grapper
    robot.set_qs(
        [1.25] * 3,
        robot.joint_name_to_indexes(
            [
                "j2n6s300_joint_finger_1",
                "j2n6s300_joint_finger_2",
                "j2n6s300_joint_finger_3",
            ]
        ),
    )

    scene = robot_scene.RobotScene(robot=robot)

    from robosim.utils import (
        PybulletDebugCtxMgr,
        plot_robot_ee_in_pybullet,
        hex_to_rgb,
        generator,
        color_qualitative,
    )

    # ctx_mgr = PybulletDebugCtxMgr(plot_robot_ee_in_pybullet(robot=robot, qs=qs))

    import numpy as np

    for demonstrations in demonstration_dataset:
        _ids = []
        color_gen = generator(color_qualitative)

        trajs = []
        for i, ee_traj in enumerate(demonstrations):
            traj = ee_traj.xs
            if traj.shape[1] == 2:
                traj = np.hstack([traj, np.zeros([traj.shape[0], 1])])
            trajs.append(traj)
        _ids.append(
            plot_robot_ee_in_pybullet(
                robot=robot,
                ee_xs=np.stack(trajs),
                traj_color=hex_to_rgb(next(color_gen)),
                point_size=5,
            )
        )

        with PybulletDebugCtxMgr(*_ids):
            for i, ee_traj in enumerate(demonstrations):
                tid = pu.add_text(
                    f"{demonstrations.name}: traj {i + 1} / {len(demonstrations)}",
                    position=[0.15, 0, 0.75],
                )

                traj = ee_traj.to_configuration_space(
                    robot=robot,
                    z_value=0.02,  # 20cm
                )

                scene.play(
                    traj,
                    robot.target_joint_names,
                    interpolate_step=args.interpolate_step,
                    delay_between_interpolated_joint=args.delay_between_interpolated_joint,
                    delay_between_joint=args.delay_between_joint,
                )
                time.sleep(args.delay_between_solution)
                pu.remove_debug(tid)

    scene.clear()
    robot.destroy()
