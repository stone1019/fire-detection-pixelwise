cmake_minimum_required(VERSION 3.0.0)
project(fire_detection VERSION 0.1.0)

include(CTest)
enable_testing()

include_directories(include)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(fireDetection main.cpp src/fire_detection.cpp)
target_link_libraries(fireDetection ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
