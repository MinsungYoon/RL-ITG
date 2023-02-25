cmake_minimum_required(VERSION 3.0.2)
project(pirl)

add_definitions(-std=c++14 -O3)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        moveit_core
        message_generation
        sensor_msgs
        moveit_msgs
        std_msgs
        moveit_ros_planning
        moveit_ros_planning_interface
        tf
        tf2
        pirl_msgs
        trac_ik_lib
        )

find_package(Boost REQUIRED)

catkin_package(
        #        INCLUDE_DIRS include
        #        LIBRARIES ${PROJECT_NAME}
        #        CATKIN_DEPENDS message_runtime
        #        DEPENDS
)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
        src/collision_checker.cpp
        src/kinematic_solver.cpp
        )
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}
        ${catkin_LIBRARIES}
        )

add_executable(pirl_services_main src/pirl_services_main.cpp)
add_dependencies(pirl_services_main ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(pirl_services_main
        ${catkin_LIBRARIES}
        ${PROJECT_NAME}
        )
