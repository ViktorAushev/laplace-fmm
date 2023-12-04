cmake_minimum_required(VERSION 3.26.3)

project(FMM VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda -use_fast_math --extra-device-vectorization --expt-relaxed-constexpr")

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)

FILE(GLOB fmmlib_src ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cpp)

add_library(fmmlib ${fmmlib_src})
target_include_directories(fmmlib PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fmmlib TBB::tbb TBB::tbbmalloc OpenMP::OpenMP_CXX MPI::MPI_CXX)

FILE(GLOB cufmmlib_src ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cu ${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cu
				 ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cpp)
add_library(cufmmlib ${cufmmlib_src})
target_include_directories(cufmmlib PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cufmmlib TBB::tbb TBB::tbbmalloc MPI::MPI_CXX)
set_target_properties(cufmmlib PROPERTIES CUDA_SEPARABLE_COMPILATION ON
										  CUDA_RESOLVE_DEVICE_SYMBOLS ON
										  CUDA_ARCHITECTURES native)


add_executable(FMM ${CMAKE_CURRENT_SOURCE_DIR}/src/examples/main.cpp)
target_link_libraries(FMM PUBLIC cufmmlib fmmlib)
