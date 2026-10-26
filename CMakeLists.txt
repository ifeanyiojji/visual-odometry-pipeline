cmake_minimum_required(VERSION 3.15)
project(mono_vo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

add_executable(mono_vo src/main.cpp)
target_include_directories(mono_vo PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(mono_vo PRIVATE ${OpenCV_LIBS} Ceres::ceres)

if(MSVC)
  target_compile_options(mono_vo PRIVATE /W4 /permissive-)
else()
  target_compile_options(mono_vo PRIVATE -Wall -Wextra -Wpedantic)
endif()
