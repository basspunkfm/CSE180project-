cmake_minimum_required(VERSION 3.8)
project(human_detection)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Create executable for human detector
add_executable(humanDetector src/humanDetector.cpp)

# Specify dependencies for the executable
ament_target_dependencies(humanDetector
  rclcpp
  rclcpp_action
  nav_msgs
  sensor_msgs
  geometry_msgs
  nav2_msgs
  std_msgs
)

# Install the executable
install(TARGETS
  humanDetector
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files (if you have any)
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

# Install config files (if you have any)
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()