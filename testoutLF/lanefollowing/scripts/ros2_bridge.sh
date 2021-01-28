#!/usr/bin/env bash
set -e

unset ROS_DISTRO
source "/opt/ros/$ROS2_DISTRO/setup.bash"
source "$LANE_FOLLOWING_ROOT_DIR/ros2_ws/install/setup.bash"

rosbridge
