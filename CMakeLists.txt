cmake_minimum_required(VERSION 3.0.2)
project(cont2)

#set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_BUILD_TYPE "Debug")
# Enable debug flags (use if you want to debug in gdb)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wuninitialized")

add_definitions(-DPJSRCDIR=\"${PROJECT_SOURCE_DIR}/\")

#find_package(OpenMP QUIET)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

find_package(catkin REQUIRED COMPONENTS
        geometry_msgs
        nav_msgs
        sensor_msgs
        roscpp
        rospy
        rosbag
        std_msgs
        tf
        eigen_conversions
        message_generation
        )

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(PkgConfig REQUIRED)
pkg_check_modules(glog REQUIRED libglog)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
#find_package(glog REQUIRED)

find_package(OpenCV 4 REQUIRED)
#find_package(OpenCV REQUIRED)
#find_package(OpenCV REQUIRED PATHS /home/lewis/.local/install_dir/opencv3.4.6/share NO_DEFAULT_PATH)

find_package(Ceres REQUIRED)

include_directories(
        include
        thirdparty
        ${glog_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${PCL_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
)

#link_directories(
#	include
#    ${PCL_LIBRARY_DIRS}
#)


catkin_package(
        CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy rosbag std_msgs message_runtime
        DEPENDS EIGEN3 PCL
        INCLUDE_DIRS include
)

# contour library
add_library(${PROJECT_NAME}contops
        src/cont2/contour.cpp
        src/cont2/contour_mng.cpp
        src/cont2/contour_db.cpp
        )

target_link_libraries(${PROJECT_NAME}contops
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${glog_LIBRARIES}
        )

### 0. rosbag play test of the project
add_executable(${PROJECT_NAME}_bag_play_test
        src/test/bag_play_test.cpp
        )
target_link_libraries(${PROJECT_NAME}_bag_play_test
        ${PROJECT_NAME}contops
        ${glog_LIBRARIES}
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${CERES_LIBRARIES}
        ${OpenCV_LIBS}
        )

### 1.1 dummy test of the project
add_executable(${PROJECT_NAME}_connect_test
        src/test/connected_comp_test.cpp
        )
target_link_libraries(${PROJECT_NAME}_connect_test
        #        ${PROJECT_NAME}map
        ${glog_LIBRARIES}
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${OpenCV_LIBS}
        )

### 1.2 test flann
add_executable(${PROJECT_NAME}_nanokdt_test
        src/test/nanoflann_test.cpp
        )
target_link_libraries(${PROJECT_NAME}_nanokdt_test
        ${glog_LIBRARIES}
        ${catkin_LIBRARIES}
        )

### 1.3 test read bin kitti
add_executable(${PROJECT_NAME}_kitti_bin_test
        src/test/kitti_read_bin_test.cpp
        )
target_link_libraries(${PROJECT_NAME}_kitti_bin_test
        ${PROJECT_NAME}contops
        ${glog_LIBRARIES}
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${CERES_LIBRARIES}
        ${OpenCV_LIBS}
        )

#add_executable(${PROJECT_NAME}_ros_lidar_test
#        src/test/dummy_test.cpp
#        )
#target_link_libraries(${PROJECT_NAME}_ros_lidar_test
#        #        ${PROJECT_NAME}map
#        ${glog_LIBRARIES}
#        ${catkin_LIBRARIES}
#        ${PCL_LIBRARIES}
#        ${OpenCV_LIBS}
#        )

