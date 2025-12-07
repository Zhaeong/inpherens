cmake_minimum_required(VERSION 3.17)
project(inpherens)

set(CMAKE_CXX_STANDARD 17)

# --- Configuration ---
# You can set this variable via command line: -DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime
if(NOT DEFINED ONNXRUNTIME_ROOT_DIR)
    message(FATAL_ERROR "Please define ONNXRUNTIME_ROOT_DIR pointing to your ONNX Runtime installation.")
endif()

# Set paths based on the root dir
set(ORT_INCLUDE_DIR "${ONNXRUNTIME_ROOT_DIR}/include")
set(ORT_LIB_DIR "${ONNXRUNTIME_ROOT_DIR}/lib")

# --- Headers ---
include_directories(${ORT_INCLUDE_DIR})

# --- Executable ---
add_executable(inpherens main.cpp)

# --- Linking ---
# Find the library file (dynamic library)
# On Windows it looks for onnxruntime.lib, on Linux libonnxruntime.so, on Mac libonnxruntime.dylib
find_library(ORT_LIB 
    NAMES onnxruntime 
    PATHS ${ORT_LIB_DIR}
    NO_DEFAULT_PATH
)

if(NOT ORT_LIB)
    message(FATAL_ERROR "Could not find onnxruntime library in ${ORT_LIB_DIR}")
endif()

target_link_libraries(inpherens PRIVATE ${ORT_LIB})

# Linux/Mac often require standard math/pthread libs
if(UNIX)
    target_link_libraries(inpherens PRIVATE pthread dl)
endif()

# On Windows, you might need to copy the DLL to the executable folder to run it.