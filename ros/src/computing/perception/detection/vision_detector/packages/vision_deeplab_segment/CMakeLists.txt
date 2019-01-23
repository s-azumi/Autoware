cmake_minimum_required(VERSION 2.8.3)
project(vision_deeplab_segment)

find_package(catkin REQUIRED COMPONENTS
        sensor_msgs
        cv_bridge
        rospy
        std_msgs
        )

catkin_package()
