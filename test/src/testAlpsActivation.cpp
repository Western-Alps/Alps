#include "testAlpsActivation.h"
#include <iostream>
#include <memory>
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
// ITK
#include "ITKHeaders.h"
#include "AlpsActivations.h"

//using ::testing::Return;

ActivationTest::ActivationTest() {
    // Have qux return true by default
    //ON_CALL(m_bar,qux()).WillByDefault(Return(true));
    // Have norf return false by default
    //ON_CALL(m_bar,norf()).WillByDefault(Return(false));
}

ActivationTest::~ActivationTest() {};

void ActivationTest::SetUp() {};

void ActivationTest::TearDown() {};
// Activation function
TEST_F(ActivationTest, ByDefaultActivationTanh_f) {
  Alps::Activation_tanh< double > activation;
  //
  //
  ASSERT_NEAR( activation.f(-1) , -0.761594, 1.e-06 );
}
// Derivative of the activation function
TEST_F(ActivationTest, ByDefaultActivationTanh_df) {
  Alps::Activation_tanh< double > activation;
  //
  //
  ASSERT_NEAR( activation.df(-1), 0.4199743416, 1.e-06 );
}
// Activation function
TEST_F(ActivationTest, ByDefaultActivationSigmoid_f) {
  Alps::Activation_sigmoid< double > activation;
  //
  //
  ASSERT_NEAR( activation.f(1) , 0.7310585786, 1.e-06 );
}
// Derivative of the activation function
TEST_F(ActivationTest, ByDefaultActivationSigmoid_df) {
  Alps::Activation_sigmoid< double > activation;
  //
  //
  EXPECT_NEAR( activation.df(1), 0.196611, 1.e-06 );
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

