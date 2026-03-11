#!/bin/bash
# robotmem ROS 2 Demo — 一键运行
source /opt/ros/humble/setup.bash
pip3 install -q git+https://github.com/robotmem/robotmem.git 2>/dev/null
curl -sO https://raw.githubusercontent.com/robotmem/robotmem/main/examples/ros2_nav_demo/demo.py
ros2 service call /reset_simulation std_srvs/srv/Empty 2>/dev/null
sleep 2
python3 demo.py 25 --prefix /originbot_1
