import argparse
import time

import pybullet as p
import pybullet_tools.utils as pu

import time
from robosim.robot_scene import JointState, robot_scene
from robosim.simulator.robot_simulator import PandaRobot, KinovaRobot
from robosim.controller_output.moveit_fake_controller import MoveItFakeController
from robosim.controller_output.kinova_client import KinovaController
from robosim.controller_output.pybullet_controller import PyBulletController
from robosim.controller_output.multicast_controller import MultiCastController
from robosim.trajectory_demonstration.clfd_helloworld_demonstration import (
    ClfdHelloworldTrajectoryDataset,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d0", "--delay-between-interpolated-joint", default=0.01, type=float
)
parser.add_argument("-d1", "--delay-between-joint", default=0.0025, type=float)
parser.add_argument("-d2", "--delay-between-solution", default=0.15, type=float)
parser.add_argument("-d3", "--delay-between-scene", default=2, type=float)
parser.add_argument("-s", "--interpolate-step", default=0, type=int)
parser.add_argument("-n", "--num-solution-to-vis", default=100, type=int)
parser.add_argument("--with-real-robot", default=False, action="store_true")
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
    print(
        " >> Remember to run 'roslaunch j2n6s300_moveit_config j2n6s300_virtual_robot_demo.launch'"
    )
    print("\n\n")

    _controllers = [MoveItFakeController(), PyBulletController()]
    if args.with_real_robot:
        _controllers.append(KinovaController())
    controller = MultiCastController(controllers=_controllers)

    robot = KinovaRobot(
        device="cpu",
        # p_client=p.DIRECT,
        p_client=p.GUI,
        include_plane=False,
        setup_learnable_model=True,
        controller=controller,
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

    for demonstrations in demonstration_dataset:
        for i, ee_traj in enumerate(demonstrations):
            # subsample
            ee_traj.xs = ee_traj.xs[::10]

            tid = pu.add_text(
                f"{demonstrations.name}: traj {i + 1} / {len(demonstrations)}",
                position=[0.15, 0, 0.75],
            )
            print(ee_traj)

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
