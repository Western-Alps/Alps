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
  std::vector< double > shv1( N, 0. );
  std::vector< double > shv2( N, 0. );
  //  
  shv1[0] = 1.; shv1[1] = 2.; shv1[2] = 3.; 
  shv2[0] = 4.; shv2[1] = 5.; shv2[2] = 6.; 
  //
  //
  EXPECT_EQ( LSE.L(shv1, shv2, N), N * 9. );
}
// Derivative of the activation function
TEST_F(CostFunctionTest, ByDefaultCostFunctionTanh_df) {
  // Loss
  Alps::LeastSquarreEstimate< double > LSE;
  //
  //
  int N = 3;
  std::vector< double > shv1( N, 0. );
  std::vector< double > shv2( N, 0. );
  std::vector< double > shv3( N, 0. );
  //  
  shv1[0] = 1.; shv1[1] = 2.; shv1[2] = 3.; 
  shv2[0] = 4.; shv2[1] = 5.; shv2[2] = 6.; 
  shv3[0] = 1.; shv3[1] = 2.; shv3[2] = 3.;
  //
  double val =0.;
  for ( int i = 0 ; i < N ; i++ )
    {
//      std::cout << "dL[" << i << "] = "
//		<< LSE.dL(shv1.get(), shv2.get(), shv3.get(), N).get()[i]
//		<< std::endl;
      val += LSE.dL(shv1, shv2, shv3, N)[i];
    }
  
  //
  //
  EXPECT_EQ( val, - static_cast< double >(N * (1+2+3)) );
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

