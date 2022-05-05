cmake_minimum_required(VERSION 3.0.2)
project(cont2)

set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wuninitialized")

#add_definitions(-DPJSRCDIR=\"${PROJECT_SOURCE_DIR}/\")

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
find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(glog REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
        include
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

# map library
#add_library(${PROJECT_NAME}map
#        src/map/mapmng.cpp
#        src/map/octree.cpp
#        )

#target_link_libraries(${PROJECT_NAME}map
#        ${catkin_LIBRARIES}
#        ${PCL_LIBRARIES}
#        glog::glog
#        )

### 0. dummy test of the project
add_executable(${PROJECT_NAME}_dummy_test
        src/test/dummy_test.cpp
        )
target_link_libraries(${PROJECT_NAME}_dummy_test
        #        ${PROJECT_NAME}map
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
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${OpenCV_LIBS}
        )

