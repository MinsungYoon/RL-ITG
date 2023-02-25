cmake_minimum_required(VERSION 2.8.3)
project(torm)

#set( CMAKE_VERBOSE_MAKEFILE true )
#message( FATAL_ERROR "[CMAKE_PREFIX_PATH]: ${CMAKE_PREFIX_PATH}" )

add_definitions(-std=c++14) # for libtorch
#add_definitions(-std=c++14 -O3) # for libtorch

#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

find_package(Torch REQUIRED PATHS /root/ThirdParty/libtorch)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

message("[CMAKE_PREFIX_PATH]: ${CMAKE_PREFIX_PATH}")
message("[CMAKE_CXX_FLAGS]: ${CMAKE_CXX_FLAGS}")



find_package(catkin REQUIRED COMPONENTS
        roscpp
        moveit_core
        moveit_ros_planning
        moveit_ros_planning_interface
        trac_ik_lib
        eigen_conversions
)

find_package(Boost REQUIRED) # 1.65

catkin_package(
        INCLUDE_DIRS include
#        CATKIN_DEPENDS roscpp
#        LIBRARIES ${PROJECT_NAME} pirl_problem_generator pirl_interpolator moveit_collision_distance_field
)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
)

# [lib] pirl_problem_generator
add_library(problem_generator
        src/utils/interpolator6D.cpp
        src/utils/pirl_problem_generator.cpp
        src/utils/eval_problem_generator.cpp
)
add_dependencies(problem_generator ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(problem_generator
        ${catkin_LIBRARIES}
        ${Boost_LIBRARIES}
)

# [lib] pirl_interpolator
add_library(pirl_interpolator
        src/pirl_interpolator.cpp
)
add_dependencies(pirl_interpolator ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(pirl_interpolator
        ${catkin_LIBRARIES}
        ${TORCH_LIBRARIES}
)

# [lib] torm
add_library(${PROJECT_NAME}
        src/torm_cost.cpp
        src/torm_parameters.cpp
        src/torm_trajectory.cpp
        src/torm_optimizer.cpp
        src/torm_ik_solver.cpp
        src/torm_debug.cpp
        src/torm_problem.cpp
        src/pirl_problem.cpp
        src/utils/traj_evaluator.cpp
        src/torm_interpolator.cpp
)
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${${PROJECT_NAME}_VERSION})
target_link_libraries(${PROJECT_NAME}
        ${catkin_LIBRARIES}
)

# [exe] main (torm_main.cpp)
add_executable(torm_main src/torm_main.cpp)
add_dependencies(torm_main ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(torm_main
        ${catkin_LIBRARIES}
        ${orocos_kdl_LIBRARIES}
        ${PROJECT_NAME}
        pirl_interpolator
        ${TORCH_LIBRARIES}
)

# [exe] main_test (torm_main_test.cpp)
add_executable(torm_main_test src/torm_main_test.cpp)
add_dependencies(torm_main_test ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(torm_main_test
        ${catkin_LIBRARIES}
        ${orocos_kdl_LIBRARIES}
        ${PROJECT_NAME}
        pirl_interpolator
        ${TORCH_LIBRARIES}
        )

# [exe] main_datagen (pirl_datagen.cpp)
add_executable(pirl_datagen src/pirl_datagen.cpp)
add_dependencies(pirl_datagen ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(pirl_datagen
        ${catkin_LIBRARIES}
        ${orocos_kdl_LIBRARIES}
        ${PROJECT_NAME}
        pirl_interpolator
        problem_generator
        ${TORCH_LIBRARIES}
)

# [exe] pirl_datacheck (pirl_datachcek.cpp)
add_executable(pirl_datacheck src/pirl_datacheck.cpp)
add_dependencies(pirl_datacheck ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(pirl_datacheck
        ${catkin_LIBRARIES}
        ${orocos_kdl_LIBRARIES}
        ${PROJECT_NAME}
        pirl_interpolator
        problem_generator
        ${TORCH_LIBRARIES}
)

# [exe] evaluation (torm_evaluation.cpp)
add_executable(torm_evaluation src/torm_evaluation.cpp)
add_dependencies(torm_evaluation ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(torm_evaluation
        ${catkin_LIBRARIES}
        ${orocos_kdl_LIBRARIES}
        ${PROJECT_NAME}
        pirl_interpolator
        ${TORCH_LIBRARIES}
)
# [exe] evaluation (torm_evaluation_init.cpp)
add_executable(torm_evaluation_init src/torm_evaluation_init.cpp)
add_dependencies(torm_evaluation_init ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(torm_evaluation_init
        ${catkin_LIBRARIES}
        ${orocos_kdl_LIBRARIES}
        ${PROJECT_NAME}
        pirl_interpolator
        ${TORCH_LIBRARIES}
        )
# [exe] evaluation (torm_evaluation_init_rnd.cpp)
add_executable(torm_evaluation_init_rnd src/torm_evaluation_init_rnd.cpp)
add_dependencies(torm_evaluation_init_rnd ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(torm_evaluation_init_rnd
        ${catkin_LIBRARIES}
        ${orocos_kdl_LIBRARIES}
        ${PROJECT_NAME}
        pirl_interpolator
        ${TORCH_LIBRARIES}
        )

# [exe] torm_generate_start_conf (torm_generate_start_conf.cpp)
add_executable(torm_generate_start_conf src/torm_generate_start_conf.cpp)
add_dependencies(torm_generate_start_conf ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(torm_generate_start_conf
        ${catkin_LIBRARIES}
        ${orocos_kdl_LIBRARIES}
        ${PROJECT_NAME}
        pirl_interpolator
        ${TORCH_LIBRARIES}
        )

# [exe] pirl_problem_check (pirl_problem_check.cpp)
add_executable(pirl_problem_check src/pirl_problem_check.cpp)
add_dependencies(pirl_problem_check ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(pirl_problem_check
        ${catkin_LIBRARIES}
        ${orocos_kdl_LIBRARIES}
        ${PROJECT_NAME}
        pirl_interpolator
        problem_generator
        ${TORCH_LIBRARIES}
)

# [exe] pirl_obs_datagen (pirl_obs_datagen.cpp)
add_executable(pirl_obs_datagen src/pirl_obs_datagen.cpp)
add_dependencies(pirl_obs_datagen ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(pirl_obs_datagen
        ${catkin_LIBRARIES}
        ${orocos_kdl_LIBRARIES}
        ${PROJECT_NAME}
        pirl_interpolator
        problem_generator
        ${TORCH_LIBRARIES}
)

# [exe] pirl_obs_problem_check (pirl_obs_datacheck.cpp)
add_executable(pirl_obs_problem_check src/pirl_obs_problem_check.cpp)
add_dependencies(pirl_obs_problem_check ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(pirl_obs_problem_check
        ${catkin_LIBRARIES}
        ${orocos_kdl_LIBRARIES}
        ${PROJECT_NAME}
        pirl_interpolator
        problem_generator
        ${TORCH_LIBRARIES}
)

# [exe] pirl_problem_generation (pirl_problem_generation.cpp)
add_executable(pirl_problem_generation src/pirl_problem_generation.cpp)
add_dependencies(pirl_problem_generation ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(pirl_problem_generation
        ${catkin_LIBRARIES}
        ${orocos_kdl_LIBRARIES}
        ${PROJECT_NAME}
        pirl_interpolator
        problem_generator
        ${TORCH_LIBRARIES}
        )

# [exe] pirl_test (pirl_test.cpp)
add_executable(pirl_nn_interpolate_test src/pirl_nn_interpolate_test.cpp)
add_dependencies(pirl_nn_interpolate_test ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(pirl_nn_interpolate_test
        ${catkin_LIBRARIES}
        ${orocos_kdl_LIBRARIES}
        ${PROJECT_NAME}
        pirl_interpolator
        ${TORCH_LIBRARIES}
)

# [exe] pirl_manual_test (pirl_manual_test.cpp)
add_executable(unit_test_collision_contacts src/unit_test_collision_contacts.cpp)
add_dependencies(unit_test_collision_contacts ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(unit_test_collision_contacts
        ${catkin_LIBRARIES}
        ${orocos_kdl_LIBRARIES}
        ${PROJECT_NAME}
        pirl_interpolator
        ${TORCH_LIBRARIES}
        )

# [exe] pirl_manual_test (pirl_manual_test.cpp)
add_executable(unit_test_collision_checking src/unit_test_collision_checking.cpp)
add_dependencies(unit_test_collision_checking ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(unit_test_collision_checking
        ${catkin_LIBRARIES}
        ${orocos_kdl_LIBRARIES}
        ${PROJECT_NAME}
        pirl_interpolator
        ${TORCH_LIBRARIES}
        )

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)