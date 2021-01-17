#
# Validation
#
file( GLOB SRCS main.cxx testAlpsCVKFolds.h testAlpsCVKFolds.cpp )
add_executable(testAlpsCVKFolds ${SRCS})
target_link_libraries( testAlpsCVKFolds ${gtest_LIB_DEPENDS}
  ${gmock_BINARY_DIR}/gtest/libgtest.a
  ${gmock_BINARY_DIR}/gtest/libgtest_main.a
  ${gmock_BINARY_DIR}/libgmock.a
  ${gmock_BINARY_DIR}/libgmock_main.a
  _Alps_utils _Alps_nn ${ITK_LIBRARIES} )
#
file( GLOB SRCS main.cxx testAlpsFullSamples.h testAlpsFullSamples.cpp )
add_executable(testAlpsFullSamples ${SRCS})
target_link_libraries( testAlpsFullSamples ${gtest_LIB_DEPENDS}
  ${gmock_BINARY_DIR}/gtest/libgtest.a
  ${gmock_BINARY_DIR}/gtest/libgtest_main.a
  ${gmock_BINARY_DIR}/libgmock.a
  ${gmock_BINARY_DIR}/libgmock_main.a
  _Alps_utils _Alps_nn ${ITK_LIBRARIES} )
#
# Input file
#
file( GLOB SRCS main.cxx testAlpsLoadDataSet.h testAlpsLoadDataSet.cpp )
add_executable(testAlpsLoadDataSet ${SRCS})
target_link_libraries( testAlpsLoadDataSet ${gtest_LIB_DEPENDS}
  ${gmock_BINARY_DIR}/gtest/libgtest.a
  ${gmock_BINARY_DIR}/gtest/libgtest_main.a
  ${gmock_BINARY_DIR}/libgmock.a
  ${gmock_BINARY_DIR}/libgmock_main.a
  _Alps_utils ${ITK_LIBRARIES} )
#
# utils
# 
file( GLOB SRCS main.cxx testITKFunctions.h testITKFunctions.cpp )
add_executable(testITKFunctions ${SRCS})
target_link_libraries( testITKFunctions ${gtest_LIB_DEPENDS}
  ${gmock_BINARY_DIR}/gtest/libgtest.a
  ${gmock_BINARY_DIR}/gtest/libgtest_main.a
  ${gmock_BINARY_DIR}/libgmock.a
  ${gmock_BINARY_DIR}/libgmock_main.a
  _Alps_utils ${ITK_LIBRARIES} )
# 
file( GLOB SRCS main.cxx testAlpsActivation.h testAlpsActivation.cpp )
add_executable(testAlpsActivation ${SRCS})
target_link_libraries( testAlpsActivation ${gtest_LIB_DEPENDS}
  ${gmock_BINARY_DIR}/gtest/libgtest.a
  ${gmock_BINARY_DIR}/gtest/libgtest_main.a
  ${gmock_BINARY_DIR}/libgmock.a
  ${gmock_BINARY_DIR}/libgmock_main.a
  _Alps_utils ${ITK_LIBRARIES} )
# 
file( GLOB SRCS main.cxx testAlpsCostFunction.h testAlpsCostFunction.cpp )
add_executable(testAlpsCostFunction ${SRCS})
target_link_libraries( testAlpsCostFunction ${gtest_LIB_DEPENDS}
  ${gmock_BINARY_DIR}/gtest/libgtest.a
  ${gmock_BINARY_DIR}/gtest/libgtest_main.a
  ${gmock_BINARY_DIR}/libgmock.a
  ${gmock_BINARY_DIR}/libgmock_main.a
  _Alps_utils ${ITK_LIBRARIES} )
# 
file( GLOB SRCS main.cxx testAlpsImage.h testAlpsImage.cpp )
add_executable(testAlpsImage ${SRCS})
target_link_libraries( testAlpsImage ${gtest_LIB_DEPENDS}
  ${gmock_BINARY_DIR}/gtest/libgtest.a
  ${gmock_BINARY_DIR}/gtest/libgtest_main.a
  ${gmock_BINARY_DIR}/libgmock.a
  ${gmock_BINARY_DIR}/libgmock_main.a
  _Alps_utils ${ITK_LIBRARIES} )
# 
file( GLOB SRCS main.cxx testAlpsLayerTensors.h testAlpsLayerTensors.cpp )
add_executable(testAlpsLayerTensors ${SRCS})
target_link_libraries( testAlpsLayerTensors ${gtest_LIB_DEPENDS}
  ${gmock_BINARY_DIR}/gtest/libgtest.a
  ${gmock_BINARY_DIR}/gtest/libgtest_main.a
  ${gmock_BINARY_DIR}/libgmock.a
  ${gmock_BINARY_DIR}/libgmock_main.a
  _Alps_utils ${ITK_LIBRARIES} )
#
file( GLOB SRCS main.cxx testAlpsSubject.h testAlpsSubject.cpp )
add_executable(testAlpsSubject ${SRCS})
target_link_libraries( testAlpsSubject ${gtest_LIB_DEPENDS}
  ${gmock_BINARY_DIR}/gtest/libgtest.a
  ${gmock_BINARY_DIR}/gtest/libgtest_main.a
  ${gmock_BINARY_DIR}/libgmock.a
  ${gmock_BINARY_DIR}/libgmock_main.a
  _Alps_utils ${ITK_LIBRARIES} )
#
file( GLOB SRCS main.cxx testAlpsSubjects.h testAlpsSubjects.cpp )
add_executable(testAlpsSubjects ${SRCS})
target_link_libraries( testAlpsSubjects ${gtest_LIB_DEPENDS}
  ${gmock_BINARY_DIR}/gtest/libgtest.a
  ${gmock_BINARY_DIR}/gtest/libgtest_main.a
  ${gmock_BINARY_DIR}/libgmock.a
  ${gmock_BINARY_DIR}/libgmock_main.a
  _Alps_utils _Alps_nn ${ITK_LIBRARIES} )
#
file( GLOB SRCS main.cxx testAlpsWeightsFclCPU.h testAlpsWeightsFclCPU.cpp )
add_executable(testAlpsWeightsFclCPU ${SRCS})
target_link_libraries( testAlpsWeightsFclCPU ${gtest_LIB_DEPENDS}
  ${gmock_BINARY_DIR}/gtest/libgtest.a
  ${gmock_BINARY_DIR}/gtest/libgtest_main.a
  ${gmock_BINARY_DIR}/libgmock.a
  ${gmock_BINARY_DIR}/libgmock_main.a
  _Alps_utils _Alps_nn ${ITK_LIBRARIES} )
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
add_test( NAME testITKFunctions          COMMAND testITKFunctions      WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY}/test/data )
#
add_test( NAME testAlpsActivation        COMMAND testAlpsActivation    WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY}/test/data )
add_test( NAME testAlpsCostFunction      COMMAND testAlpsCostFunction  WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY}/test/data )
add_test( NAME testAlpsCVKFolds          COMMAND testAlpsCVKFolds      WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY}/test/data )
add_test( NAME testAlpsFullSamples       COMMAND testAlpsFullSamples   WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY}/test/data )
add_test( NAME testAlpsLoadDataSet       COMMAND testAlpsLoadDataSet   WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY}/test/data )
add_test( NAME testAlpsSubject           COMMAND testAlpsSubject       WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY}/test/data )
add_test( NAME testAlpsSubjects          COMMAND testAlpsSubjects      WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY}/test/data )
add_test( NAME testAlpsImage             COMMAND testAlpsImage         WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY}/test/data )
add_test( NAME testAlpsLayerTensors      COMMAND testAlpsLayerTensors  WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY}/test/data )
add_test( NAME testAlpsWeightsFclCPU     COMMAND testAlpsWeightsFclCPU WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY}/test/data )
add_test( NAME testAlpsConvolutionWindow COMMAND testAlpsConvolutionWindow )
