cmake_minimum_required(VERSION 3.15)

project(Fast_Projective_Skinning)

# set output paths
set(PROJECT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

# compile PMP library
set(PMP_BUILD_APPS     OFF CACHE BOOL "")
set(PMP_BUILD_EXAMPLES OFF CACHE BOOL "")
set(PMP_BUILD_TESTS    OFF CACHE BOOL "")
set(PMP_BUILD_DOCS     OFF CACHE BOOL "")
add_subdirectory(external/pmp-library)

# add include directories
include_directories(${PROJECT_SOURCE_DIR}/external/pmp-library/src)
include_directories(${PROJECT_SOURCE_DIR}/external/pmp-library/external/imgui)
include_directories(${PROJECT_SOURCE_DIR}/external/pmp-library/external/glfw/include)
include_directories(${PROJECT_SOURCE_DIR}/external/pmp-library/external/glew/include)

# set default compiler flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if((UNIX OR APPLE) AND CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -msse2 -std=c++14 -pedantic -Wall -Wextra")
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang" AND NOT EMSCRIPTEN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pedantic -Wall -Wextra -Wno-unused-parameter")
endif()

if(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_MATH_DEFINES -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /wd4267 /wd4244 /wd4305") # conversion/truncation from size_t and double to float
endif()

if (EMSCRIPTEN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -s ALLOW_MEMORY_GROWTH=1 -s USE_GLFW=3 -s WASM=0 -s USE_WEBGL2=1")
    set(CMAKE_EXECUTABLE_SUFFIX ".html")
endif()

# Eigen
set(EIGEN_SOURCE_DIR "external/eigen")
include_directories(${EIGEN_SOURCE_DIR})

# openmp
find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS} -fopenmp)
endif()

# CUDA
include(CheckLanguage)
check_language(CUDA)
if(DEFINED CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  message("CUDA FOUND ${CMAKE_CUDA_COMPILER_VERSION}")
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -w -Xcompiler -fopenmp)
  add_compile_options(-D WITH_CUDA)
else()
  message("CUDA NOT FOUND")
endif()

# make a release build by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()


if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb")
endif()


# which directories to process
add_subdirectory(src/)
