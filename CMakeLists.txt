cmake_minimum_required(VERSION 2.8)

project(Alps)

#
# Compilation

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_definitions(/DYOURDEFINITION)
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
  add_definitions(-std=c++11)
  set(CMAKE_CXX_FLAGS "-O3")
endif()

#
# Eigen
set( EIGEN /mnt/neuroimaging/Eigen-3/install/include/eigen3/ /mnt/neuroimaging/Eigen-3/install/include/eigen3/unsupported/ )
set( JSON /mnt/neuroimaging/json/src/ )
#
include_directories( ${EIGEN}/ ${JSON}/ ${ITK_INC})
list( APPEND SOURCES ${DLIB_ALL_SOURCES} )



#
# ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

#
# project
include_directories( ${Alps_SOURCE_DIR}  )
#

#
file( GLOB SOURCES_ *.cxx )
list( APPEND SOURCES ${SOURCES_} )
#message(STATUS ${SOURCES})

#
# Glbal
file( GLOB HEADERS_ utils/*.h )
file( GLOB SOURCES_ utils/*.cxx )
#
list( APPEND HEADERS ${HEADERS_}  )
list( APPEND SOURCES ${SOURCES_} )
include_directories( ${Alps_SOURCE_DIR}/utils  )

#
# CNN
#
# Monte Rosa
file( GLOB Monte_Rosa_HEADERS_ Monte_Rosa/*.h )
file( GLOB Monte_Rosa_SOURCES_ Monte_Rosa/*.cxx )
#
list( APPEND HEADERS ${Monte_Rosa_HEADERS_} )
list( APPEND SOURCES ${Monte_Rosa_SOURCES_} )
#
add_executable( Monte_Rosa MACOSX_BUNDLE Monte_Rosa/Monte_Rosa.cpp ${SOURCES} )
# cmake 3.1 target_compile_features(Monte_Rosa PRIVATE cxx_range_for)
target_link_libraries( Monte_Rosa -lX11 ${ITK_LIBRARIES} ${HEADER_FILES} )

#
# ???
#
# Mont Blanc
file( GLOB Mont_Blanc_HEADERS_ Mont_Blanc/*.h )
file( GLOB Mont_Blanc_SOURCES_ Mont_Blanc/*.cxx )
#
list( APPEND HEADERS ${Mont_Blanc_HEADERS_} )
list( APPEND SOURCES ${Mont_Blanc_SOURCES_} )
#
add_executable( Mont_Blanc MACOSX_BUNDLE Mont_Blanc/Mont_Blanc.cpp ${SOURCES} )
# cmake 3.1 target_compile_features(Mont_Blanc PRIVATE cxx_range_for)
target_link_libraries( Mont_Blanc -lX11 ${ITK_LIBRARIES} ${HEADER_FILES} )


#
# Install
install( FILES ${HEADERS_} ${Monte_Rosa_HEADERS_} ${Mont_Blanc_HEADERS_} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/ )
