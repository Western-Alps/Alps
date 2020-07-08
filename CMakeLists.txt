include(FetchContent)
cmake_minimum_required(VERSION 3.2)

project(Alps)

#
# CUDA
find_package(CUDA QUIET REQUIRED)
include(FindCUDA)
#
set(CUDA_SEPARABLE_COMPILATION ON)
#
LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -O3 -std=c++11 -lineinfo -use_fast_math -Xptxas -dlcm=cg -Wno-deprecated-gpu-targets )
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_37,code=sm_37)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)



#
# Compilation
SET(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
SET(BUILD_SHARED_LIBRARIES OFF)
#SET(CMAKE_EXE_LINKER_FLAGS "-static")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_definitions(/DYOURDEFINITION)
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
  add_definitions(-std=c++11)
  set(CMAKE_CXX_FLAGS "-O3 -ffast-math -Wall")
endif()

#
# Eigen and JSON
set( EIGEN /mnt/neuroimaging/Eigen-3/install/include/eigen3/ /mnt/neuroimaging/Eigen-3/install/include/eigen3/unsupported/ )
set( JSON /mnt/neuroimaging/json/src/ )

#
# include directories
include_directories( /usr/local/cuda/include ${EIGEN}/ ${JSON}/ ${ITK_INC} )
list( APPEND SOURCES ${DLIB_ALL_SOURCES} )



#
# ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

#
# project
include_directories( ${Alps_SOURCE_DIR}  )
#

##
#file( GLOB SOURCES_ *.cxx  )
#list( APPEND SOURCES ${SOURCES_} )
##message(STATUS ${SOURCES})

#
# Global
#file(  NN_HEADERS_
#  NN/Functions/*.h NN/*.cuh NN/*.h
#  ITK/Modules/Core/ImageFunction/include/*.h )
#file(  UTILS_HEADERS_
#  utils/*.h  )
file( GLOB HEADERS_
  utils/*.h utils/CrossValidation/*.h
  NN/Functions/*.h NN/Gradients/*.h NN/*.cuh NN/*.h
  ITK/Modules/Core/ImageFunction/include/*.h )
#
file( GLOB NN_SOURCES_
  NN/*.cu  NN/*.cxx 
  ITK/Modules/Core/ImageFunction/include/*.hxx )
file( GLOB UTILS_SOURCES_
  utils/*.cxx  )
#file( GLOB SOURCES_
#  utils/*.cxx NN/*.cu  NN/*.cxx 
#  ITK/Modules/Core/ImageFunction/include/*.hxx )

list( APPEND HEADERS ${HEADERS_}  )
list( APPEND SOURCES ${NN_SOURCES_} ${UTILS_SOURCES_} )
#list( APPEND HEADERS ${HEADERS_}  )
#list( APPEND SOURCES ${SOURCES_} )
include_directories(
  ${Alps_SOURCE_DIR}/utils
  ${Alps_SOURCE_DIR}/utils/CrossValidation
  ${Alps_SOURCE_DIR}/NN
  ${Alps_SOURCE_DIR}/NN/Functions
  ${Alps_SOURCE_DIR}/NN/Gradients
  ${Alps_SOURCE_DIR}/ITK/Modules/Core/ImageFunction/include
   )

#
# Libraries
cuda_add_library( _Alps_nn SHARED ${NN_SOURCES_} OPTIONS --compiler-options "-fPIC" )
add_library( _Alps_utils SHARED ${UTILS_SOURCES_} )
#
#link_directories (${LIB_BINARY_DIR}/NN ${LIB_BINARY_DIR}/utils ) 

#
# CNN + DC
#
# Monte Rosa
include_directories( ${Alps_SOURCE_DIR}/Monte_Rosa )
#
file( GLOB Monte_Rosa_HEADERS_ Monte_Rosa/*.h Monte_Rosa/*.cuh )
file( GLOB Monte_Rosa_SOURCES_ Monte_Rosa/*.cxx Monte_Rosa/*.cu )
#
list( APPEND HEADERS ${Monte_Rosa_HEADERS_} )
list( APPEND SOURCES ${Monte_Rosa_SOURCES_} )
#
cuda_add_executable( Monte_Rosa MACOSX_BUNDLE Monte_Rosa/Monte_Rosa.cpp ${HEADERS} ${Monte_Rosa_SOURCES_} )
# cmake 3.1 target_compile_features(Monte_Rosa PRIVATE cxx_range_for)
target_link_libraries( Monte_Rosa ${ITK_LIBRARIES} _Alps_nn _Alps_utils )
#
#
# Auto-encoder
#
# Mont Blanc
include_directories( ${Alps_SOURCE_DIR}/Mont_Blanc )
#
file( GLOB Mont_Blanc_HEADERS_ Mont_Blanc/*.h  Mont_Blanc/*.cuh )
file( GLOB Mont_Blanc_SOURCES_ Mont_Blanc/*.cxx  Mont_Blanc/*.cxx )
#
list( APPEND HEADERS ${Mont_Blanc_HEADERS_} )
list( APPEND SOURCES ${Mont_Blanc_SOURCES_} )
#
cuda_add_executable( Mont_Blanc MACOSX_BUNDLE Mont_Blanc/Mont_Blanc.cpp ${HEADERS} ${Mont_Blanc_SOURCES_} )
# cmake 3.1 target_compile_features(Mont_Blanc PRIVATE cxx_range_for)
target_link_libraries( Mont_Blanc  ${ITK_LIBRARIES} _Alps_nn _Alps_utils )
#
# Mont Maudi
include_directories( ${Alps_SOURCE_DIR}/Mont_Maudit )
#
file( GLOB Mont_Maudit_HEADERS_ Mont_Maudit/*.h  Mont_Maudit/*.cuh )
file( GLOB Mont_Maudit_SOURCES_ Mont_Maudit/*.cxx  Mont_Maudit/*.cxx )
#
list( APPEND HEADERS ${Mont_Maudit_HEADERS_} )
list( APPEND SOURCES ${Mont_Maudit_SOURCES_} )
#
cuda_add_executable( Mont_Maudit MACOSX_BUNDLE Mont_Maudit/Mont_Maudit.cpp  ${HEADERS} ${Mont_Maudit_SOURCES_} )
# cmake 3.1 target_compile_features(Mont_Maudit PRIVATE cxx_range_for)
target_link_libraries( Mont_Maudit  ${ITK_LIBRARIES}  _Alps_nn _Alps_utils )


#
# Install
install( FILES ${HEADERS_} ${Monte_Rosa_HEADERS_} ${Mont_Blanc_HEADERS_}  ${Mont_Maudit_HEADERS_} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/ )

#
# Goolge tests
enable_testing()
add_subdirectory(test)
