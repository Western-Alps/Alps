#
#
file(GLOB SRCS main.cxx testLoadDataSet.h testLoadDataSet.cpp)
#
add_executable(testLoadDtataSet ${SRCS})
target_link_libraries( testLoadDtataSet ${gtest_LIB_DEPENDS}
  ${gmock_BINARY_DIR}/gtest/libgtest.a
  ${gmock_BINARY_DIR}/gtest/libgtest_main.a
  ${gmock_BINARY_DIR}/libgmock.a ${gmock_BINARY_DIR}/libgmock_main.a
  _Alps_utils ${ITK_LIBRARIES} )
add_test( NAME testLoadDtataSet COMMAND testLoadDtataSet WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY}/test/data )
