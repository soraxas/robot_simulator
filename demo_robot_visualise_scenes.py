import argparse
import time


from robosim.scene import Trajectory, JointState, motion_bench_scene, robot_scene
from robosim.simulator.robot_simulator import KinovaRobot, PandaRobot
from robosim.utils import get_project_root
import pybullet as p

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d0", "--delay-between-interpolated-joint", default=0.01, type=float
)
parser.add_argument("-d1", "--delay-between-joint", default=0.2, type=float)
parser.add_argument("-d2", "--delay-between-solution", default=0.5, type=float)
parser.add_argument("-d3", "--delay-between-scene", default=2, type=float)
parser.add_argument("-s", "--interpolate-step", default=50, type=int)
parser.add_argument(
    "-t", "--tags-to-vis", default="all", choices=["all"] + motion_bench_scene.tag_names
)
parser.add_argument("-n", "--num-solution-to-vis", default=100, type=int)
parser.add_argument(
    "-r",
    "--visualise-request",
    help="visualise the request start and target directly, "
    "instead of the example trajectory",
    action="store_true",
)

if __name__ == "__main__":
    project_path = get_project_root()

    args = parser.parse_args()

    if args.tags_to_vis == "all":
        args.tags_to_vis = list(motion_bench_scene.tag_names)
    else:
        args.tags_to_vis = args.tags_to_vis.split(",")
    print("\n\n")

    ###################################################################################################
    # playing self-created trajectory
    ###################################################################################################

    with KinovaRobot(
        device="cpu",
        p_client=p.GUI,
        include_plane=False,
        setup_learnable_model=True,
    ) as kinova_robot:
        kinova_robot.print_info()
        with robot_scene.RobotScene(robot=kinova_robot) as scene:
            traj = Trajectory(
                [
                    JointState(
                        name=[
                            "j2n6s300_joint_3",
                            "j2n6s300_joint_4",
                            "j2n6s300_joint_5",
                        ],
                        position=[0, 0, 0],
                    ),
                    JointState(
                        name=[
                            "j2n6s300_joint_3",
                            "j2n6s300_joint_4",
                            "j2n6s300_joint_5",
                        ],
                        position=[1, 1, 1],
                    ),
                    JointState(
                        name=[
                            "j2n6s300_joint_3",
                            "j2n6s300_joint_4",
                            "j2n6s300_joint_5",
                        ],
                        position=[2, 2, 2],
                    ),
                    JointState(
                        name=[
                            "j2n6s300_joint_3",
                            "j2n6s300_joint_4",
                            "j2n6s300_joint_5",
                        ],
                        position=[2, 2, 5],
                    ),
                ]
            )
            scene.play(
                traj,
                delay_between_joint=0.1,
            )

    ###################################################################################################
    # playing motion bench
    ###################################################################################################

    with PandaRobot(
        device="cpu",
        p_client=p.GUI,
        include_plane=False,
        setup_learnable_model=True,
    ) as panda_robot:
        for tag_name in args.tags_to_vis:
            print("=" * 40)

            with motion_bench_scene.MotionBenchScene(
                robot=panda_robot, tag_name=tag_name
            ) as scene:
                print(f"Scene: {tag_name}\n")
                with scene:
                    for i, (request_fn, traj_fn) in enumerate(
                        zip(scene.request_paths, scene.trajectory_paths)
                    ):
                        if i >= args.num_solution_to_vis:
                            break

                        print("-" * 20)
                        request = motion_bench_scene.path_request_from_yaml(request_fn)

                        if args.visualise_request:
                            traj = Trajectory(
                                [request.start_state, request.target_state]
                            )
                        else:
                            traj = motion_bench_scene.trajectory_from_yaml(traj_fn)

                        print(f"Path Request:\n{request}\n")
                        print(f"Example Trajectory:\n{traj}\n")

                        import pybullet_tools.utils as pu

                        tid = pu.add_text(
                            f"{tag_name}: traj {i + 1} / {min(len(scene), args.num_solution_to_vis)}",
                            position=[0, 0, 1.5],
                        )
                        scene.play(
                            traj,
                            panda_robot.target_joint_names,
                            interpolate_step=args.interpolate_step,
                            delay_between_interpolated_joint=args.delay_between_interpolated_joint,
                            delay_between_joint=args.delay_between_joint,
                        )

                        time.sleep(args.delay_between_solution)
                        pu.remove_debug(tid)

            time.sleep(args.delay_between_scene)
