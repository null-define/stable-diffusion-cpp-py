cmake_minimum_required(VERSION 3.5...3.26)
set(CMAKE_CXX_STANDARD 17)

project(sd_cpp_lib LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS true)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(
  Python
  COMPONENTS Interpreter Development
  REQUIRED)

add_subdirectory(./pybind11 EXCLUDE_FROM_ALL)
add_subdirectory(./stable-diffusion.cpp EXCLUDE_FROM_ALL)

set(TARGET sd_cpp_lib)
pybind11_add_module(${TARGET} sd_cpp_lib.cc)
target_link_libraries(${TARGET} PRIVATE stable-diffusion ggml)


