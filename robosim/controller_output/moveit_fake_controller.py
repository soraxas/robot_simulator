import copy

import rospy

from typing import List
from sensor_msgs.msg import JointState


from robosim.controller_output import AbstractController
from robosim.simulator.data_types import ConfigurationSpaceType
from robosim.controller_output import ControllerJointType, AbstractController


class MoveItFakeController(AbstractController):
    controller_input_type: ControllerJointType = ControllerJointType.joint_name

    def __init__(
        self, joint_state_topic: str = "/move_group/fake_controller_joint_states"
    ) -> None:
        self.joint_state_topic = joint_state_topic
        self.pub = rospy.Publisher(self.joint_state_topic, JointState, queue_size=10)
        self.msg_template = self._build_msg_template()

        rospy.init_node(self.__class__.__name__, anonymous=False)

    def _build_msg_template(self) -> JointState:
        msg_template = JointState()
        msg_template.header.frame_id = "world"
        return msg_template

    def get_msg_template(self) -> JointState:
        msg = copy.copy(self.msg_template)
        msg.header.stamp = rospy.Time.now()
        return msg

    def set_qs_with_names(
        self,
        qs: ConfigurationSpaceType,
        joint_names: List[str] = None,
    ):
        msg = self.get_msg_template()
        msg.name = joint_names
        msg.position = qs

        # print(qs, joint_names)
        # print(msg)
        self.pub.publish(msg)
