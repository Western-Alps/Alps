#
# Basics
file( GLOB SRCS main.cxx testLoadDataSet.h testLoadDataSet.cpp )
add_executable(testLoadDataSet ${SRCS})
target_link_libraries( testLoadDataSet ${gtest_LIB_DEPENDS}
  ${gmock_BINARY_DIR}/gtest/libgtest.a
  ${gmock_BINARY_DIR}/gtest/libgtest_main.a
  ${gmock_BINARY_DIR}/libgmock.a
  ${gmock_BINARY_DIR}/libgmock_main.a
  _Alps_utils ${ITK_LIBRARIES} )
#
file( GLOB SRCS main.cxx  testAlpsTensor.h testAlpsTensor.cpp)
add_executable(testAlpsTensor ${SRCS})
target_link_libraries(testAlpsTensor  ${gtest_LIB_DEPENDS} 
  ${gmock_BINARY_DIR}/gtest/libgtest.a
  ${gmock_BINARY_DIR}/gtest/libgtest_main.a
  ${gmock_BINARY_DIR}/libgmock.a
  ${gmock_BINARY_DIR}/libgmock_main.a
  _Alps_utils ${ITK_LIBRARIES} )
#
# Convolution
file( GLOB SRCS main.cxx testAlpsConvolutionWindow.h testAlpsConvolutionWindow.cpp )
add_executable(testAlpsConvolutionWindow ${SRCS})
target_link_libraries(testAlpsConvolutionWindow  ${gtest_LIB_DEPENDS}  
  ${gmock_BINARY_DIR}/gtest/libgtest.a
  ${gmock_BINARY_DIR}/gtest/libgtest_main.a
  ${gmock_BINARY_DIR}/libgmock.a
  ${gmock_BINARY_DIR}/libgmock_main.a
  _Alps_utils ${ITK_LIBRARIES} )


#
# Tests
add_test( NAME testLoadDataSet COMMAND testLoadDataSet WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY}/test/data )
add_test( NAME testAlpsTensor COMMAND testAlpsTensor )
add_test( NAME testAlpsConvolutionWindow COMMAND testAlpsConvolutionWindow )
