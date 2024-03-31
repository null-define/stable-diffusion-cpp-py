cmake_minimum_required(VERSION 3.21)

project(stable_diffusion_cpp)

add_subdirectory(sd_cpp_lib)

install(
  TARGETS sd_cpp_lib
  LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/stable_diffusion_cpp
  RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/stable_diffusion_cpp
  ARCHIVE DESTINATION ${SKBUILD_PLATLIB_DIR}/stable_diffusion_cpp
  FRAMEWORK DESTINATION ${SKBUILD_PLATLIB_DIR}/stable_diffusion_cpp
  RESOURCE DESTINATION ${SKBUILD_PLATLIB_DIR}/stable_diffusion_cpp)
