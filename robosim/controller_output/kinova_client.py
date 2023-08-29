import rospy
import actionlib

import kinova_msgs.msg
import geometry_msgs.msg
import std_msgs.msg
from kinova_msgs.srv import *
from sensor_msgs.msg import JointState
import argparse


from typing import List
from sensor_msgs.msg import JointState


from robosim.controller_output.ros_controller import RosController
from robosim.simulator.data_types import ConfigurationSpaceType
from robosim.controller_output import ControllerJointType, AbstractController


def kinova_joint_position_client(angle_set: List[float], prefix: str = "j2n6s300"):
    action_address = f"/{prefix}driver/joints_action/joint_angles"
    client = actionlib.SimpleActionClient(
        action_address, kinova_msgs.msg.ArmJointAnglesAction
    )
    if not client.wait_for_server(rospy.Duration(1.0)):
        print(
            f">> action server time-outed. Have you run\n>> 'roslaunch kinova_bringup kinova_robot.launch kinova_robotType:={prefix}'?"
        )
        exit(1)

    goal = kinova_msgs.msg.ArmJointAnglesGoal()
    goal.angles.joint1 = angle_set[0]
    goal.angles.joint2 = angle_set[1]
    goal.angles.joint3 = angle_set[2]
    goal.angles.joint4 = angle_set[3]
    goal.angles.joint5 = angle_set[4]
    goal.angles.joint6 = angle_set[5]
    goal.angles.joint7 = angle_set[6]

    client.send_goal(goal)

    assert client.wait_for_result(rospy.Duration(100.0))

    # Prints out the result of executing the action
    return client.get_result()


class KinovaController(RosController):
    controller_input_type: ControllerJointType = ControllerJointType.joint_index

    def __init__(
        self, joint_state_topic: str = "/move_group/fake_controller_joint_states"
    ) -> None:
        super().__init__()
        # self.joint_state_topic = joint_state_topic
        # self.pub = rospy.Publisher(self.joint_state_topic, JointState, queue_size=10)
        # self.msg_template = self._build_msg_template()

        self.index_mapper = {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 5,
        }

    def set_qs_with_indexes(
        self,
        qs: ConfigurationSpaceType,
        joint_indexes: List[int],
    ):
        angles = [0] * 7
        for index, q in zip(joint_indexes, qs):
            if index not in self.index_mapper:
                continue
            index = self.index_mapper[index]
            # only has 6 joints
            if index < 0 or index >= 6:
                print(index)
                raise NotImplementedError(
                    f"index {index} is not supported (full indices given are {joint_indexes})"
                )
            angles[index] = q
        print(kinova_joint_position_client(angle_set=angles))
