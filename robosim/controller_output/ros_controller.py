import rospy

from robosim.controller_output import AbstractController


class RosController(AbstractController):
    def __init__(self) -> None:
        try:
            RosController.__ROS_initialised
        except AttributeError:
            rospy.init_node(RosController.__name__, anonymous=False)
            RosController.__ROS_initialised = True
