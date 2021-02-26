#include "testAlpsSGD.h"
#include <iostream>
#include <memory>
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
// ITK
#include "ITKHeaders.h"
#include "AlpsGradient.h"
#include "AlpsSGD.h"

//using ::testing::Return;

SGCTest::SGCTest() {
    // Have qux return true by default
    //ON_CALL(m_bar,qux()).WillByDefault(Return(true));
    // Have norf return false by default
    //ON_CALL(m_bar,norf()).WillByDefault(Return(false));
}

SGCTest::~SGCTest() {};

void SGCTest::SetUp() {};

void SGCTest::TearDown() {};
// CostFunction function
TEST_F(SGCTest, ByDefaultSGD0) {
  //
  //
  Alps::StochasticGradientDescent< double,
				   Eigen::MatrixXd,
				   Eigen::MatrixXd,
				   Alps::Arch::CPU >  Sgd;
  //
  //
  EXPECT_EQ( 0., 0. );
}
// Derivative of the activation function
TEST_F(SGCTest, ByDefaultSGDChild) {
  //
  //
  std::shared_ptr< Alps::Gradient_base > gradient_ = std::make_shared< Alps::StochasticGradientDescent< double,
													Eigen::MatrixXd,
													Eigen::MatrixXd,
													Alps::Arch::CPU > >();
  //
  std::dynamic_pointer_cast< Alps::Gradient< Eigen::MatrixXd,
					     Eigen::MatrixXd > >(gradient_)->set_parameters( 4, 3 );

  //
  //
  Eigen::MatrixXd delta = Eigen::MatrixXd::Zero(4,1);
  Eigen::MatrixXd z     = Eigen::MatrixXd::Zero(3,1);
  //
  delta(0,0) = 1. ; delta(1,0) = 2. ; delta(2,0) = 3. ; delta(3,0) = 4. ; 
  z(0,0)     = 10.; z(1,0)     = 20.; z(2,0)     = 30.; 
  //
  Eigen::MatrixXd test = delta * z.transpose();
  std::cout << "The test: \n" << test << std::endl;
  
  //
  //
  std::dynamic_pointer_cast< Alps::Gradient< Eigen::MatrixXd,
					     Eigen::MatrixXd > >(gradient_)->add_tensors( delta, z );
  //
  Eigen::MatrixXd test2 = std::dynamic_pointer_cast< Alps::Gradient< Eigen::MatrixXd,
								     Eigen::MatrixXd > >(gradient_)->solve();
  std::cout << "The test2: \n" << test2 << std::endl;
 
  //
  //
  EXPECT_EQ( test(2,2), test2(2,2) );
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

