#include "testAlpsCostFunction.h"
#include <iostream>
#include <memory>
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
// ITK
#include "ITKHeaders.h"
#include "AlpsCostFunction.h"
#include "AlpsActivations.h"

//using ::testing::Return;

CostFunctionTest::CostFunctionTest() {
    // Have qux return true by default
    //ON_CALL(m_bar,qux()).WillByDefault(Return(true));
    // Have norf return false by default
    //ON_CALL(m_bar,norf()).WillByDefault(Return(false));
}

CostFunctionTest::~CostFunctionTest() {};

void CostFunctionTest::SetUp() {};

void CostFunctionTest::TearDown() {};
// CostFunction function
TEST_F(CostFunctionTest, ByDefaultCostFunctionLSE_L) {
  // Loss
  Alps::LeastSquarreEstimate< double > LSE;
  //
  //
  int N = 3;
  std::shared_ptr< double > shv1( new double[N], std::default_delete< double [] >() );
  std::shared_ptr< double > shv2( new double[N], std::default_delete< double [] >() );
  //  
  shv1.get()[0] = 1.; shv1.get()[1] = 2.; shv1.get()[2] = 3.; 
  shv2.get()[0] = 4.; shv2.get()[1] = 5.; shv2.get()[2] = 6.; 
  //
  //
  EXPECT_EQ( LSE.L(shv1.get(), shv2.get(), N), N * 9. );
}
// Derivative of the activation function
TEST_F(CostFunctionTest, ByDefaultCostFunctionTanh_df) {
  //
  //
  EXPECT_EQ( 0, 0 );
}

//TEST_F(ImageTest, ByDefaultBazFalseIsFalse) {
//    Image foo(m_bar);
//    EXPECT_EQ(foo.baz(false), false);
//}
//
//TEST_F(ImageTest, SometimesBazFalseIsTrue) {
//    Image foo(m_bar);
//    // Have norf return true for once
//    EXPECT_CALL(m_bar,norf()).WillOnce(Return(true));
//    EXPECT_EQ(foo.baz(false), true);
//}

