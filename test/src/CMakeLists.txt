#
# Basics
#
file( GLOB SRCS main.cxx testCVKFolds.h testCVKFolds.cpp )
add_executable(testCVKFolds ${SRCS})
target_link_libraries( testCVKFolds ${gtest_LIB_DEPENDS}
  ${gmock_BINARY_DIR}/gtest/libgtest.a
  ${gmock_BINARY_DIR}/gtest/libgtest_main.a
  ${gmock_BINARY_DIR}/libgmock.a
  ${gmock_BINARY_DIR}/libgmock_main.a
  _Alps_utils _Alps_nn ${ITK_LIBRARIES} )
#
file( GLOB SRCS main.cxx testFullSamples.h testFullSamples.cpp )
add_executable(testFullSamples ${SRCS})
target_link_libraries( testFullSamples ${gtest_LIB_DEPENDS}
  ${gmock_BINARY_DIR}/gtest/libgtest.a
  ${gmock_BINARY_DIR}/gtest/libgtest_main.a
  ${gmock_BINARY_DIR}/libgmock.a
  ${gmock_BINARY_DIR}/libgmock_main.a
  _Alps_utils _Alps_nn ${ITK_LIBRARIES} )
#
file( GLOB SRCS main.cxx testLoadDataSet.h testLoadDataSet.cpp )
add_executable(testLoadDataSet ${SRCS})
target_link_libraries( testLoadDataSet ${gtest_LIB_DEPENDS}
  ${gmock_BINARY_DIR}/gtest/libgtest.a
  ${gmock_BINARY_DIR}/gtest/libgtest_main.a
  ${gmock_BINARY_DIR}/libgmock.a
  ${gmock_BINARY_DIR}/libgmock_main.a
  _Alps_utils ${ITK_LIBRARIES} )
#
file( GLOB SRCS main.cxx testImage.h testImage.cpp )
add_executable(testImage ${SRCS})
target_link_libraries( testImage ${gtest_LIB_DEPENDS}
  ${gmock_BINARY_DIR}/gtest/libgtest.a
  ${gmock_BINARY_DIR}/gtest/libgtest_main.a
  ${gmock_BINARY_DIR}/libgmock.a
  ${gmock_BINARY_DIR}/libgmock_main.a
  _Alps_utils ${ITK_LIBRARIES} )
#
file( GLOB SRCS main.cxx testSubject.h testSubject.cpp )
add_executable(testSubject ${SRCS})
target_link_libraries( testSubject ${gtest_LIB_DEPENDS}
  ${gmock_BINARY_DIR}/gtest/libgtest.a
  ${gmock_BINARY_DIR}/gtest/libgtest_main.a
  ${gmock_BINARY_DIR}/libgmock.a
  ${gmock_BINARY_DIR}/libgmock_main.a
  _Alps_utils ${ITK_LIBRARIES} )
file( GLOB SRCS main.cxx testImage.h testImage.cpp )
#
file( GLOB SRCS main.cxx testSubjects.h testSubjects.cpp )
add_executable(testSubjects ${SRCS})
target_link_libraries( testSubjects ${gtest_LIB_DEPENDS}
  ${gmock_BINARY_DIR}/gtest/libgtest.a
  ${gmock_BINARY_DIR}/gtest/libgtest_main.a
  ${gmock_BINARY_DIR}/libgmock.a
  ${gmock_BINARY_DIR}/libgmock_main.a
  _Alps_utils _Alps_nn ${ITK_LIBRARIES} )
#
file( GLOB SRCS main.cxx testWeightsFclCPU.h testWeightsFclCPU.cpp )
add_executable(testWeightsFclCPU ${SRCS})
target_link_libraries( testWeightsFclCPU ${gtest_LIB_DEPENDS}
  ${gmock_BINARY_DIR}/gtest/libgtest.a
  ${gmock_BINARY_DIR}/gtest/libgtest_main.a
  ${gmock_BINARY_DIR}/libgmock.a
  ${gmock_BINARY_DIR}/libgmock_main.a
  _Alps_utils _Alps_nn ${ITK_LIBRARIES} )
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
add_test( NAME testCVKFolds      COMMAND testCVKFolds      WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY}/test/data )
add_test( NAME testFullSamples   COMMAND testFullSamples   WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY}/test/data )
add_test( NAME testLoadDataSet   COMMAND testLoadDataSet   WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY}/test/data )
add_test( NAME testSubject       COMMAND testSubject       WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY}/test/data )
add_test( NAME testSubjects      COMMAND testSubjects      WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY}/test/data )
add_test( NAME testImage         COMMAND testImage         WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY}/test/data )
add_test( NAME testWeightsFclCPU COMMAND testWeightsFclCPU WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY}/test/data )
add_test( NAME testAlpsTensor    COMMAND testAlpsTensor )
add_test( NAME testAlpsConvolutionWindow COMMAND testAlpsConvolutionWindow )