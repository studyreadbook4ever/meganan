cmake_minimum_required(VERSION 3.20)
project(meganan LANGUAGES CXX)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
find_package(pybind11 3.0 CONFIG REQUIRED PATHS "${PYBIND11_CMAKE_DIR}" NO_DEFAULT_PATH)
find_package(JPEG REQUIRED)
find_package(Threads REQUIRED)

pybind11_add_module(_motion_native NO_EXTRAS
    native/module.cpp native/camera.cpp native/inference.cpp native/features.cpp native/movenet.cpp)
target_compile_features(_motion_native PRIVATE cxx_std_17)
target_compile_options(_motion_native PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(_motion_native PRIVATE native)
target_include_directories(_motion_native SYSTEM PRIVATE native/third_party/mediapipe native/third_party/litert)
target_link_libraries(_motion_native PRIVATE JPEG::JPEG Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(_motion_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python")
