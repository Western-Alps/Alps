#include "testAlpsConvolutionWindow.h"
#include "AlpsTensor.h"
#include "AlpsWindow.h"

//using ::testing::Return;

AlpsConvolutionWindowTest::AlpsConvolutionWindowTest() {
    // Have qux return true by default
    //ON_CALL(m_bar,qux()).WillByDefault(Return(true));
    // Have norf return false by default
    //ON_CALL(m_bar,norf()).WillByDefault(Return(false));
}

AlpsConvolutionWindowTest::~AlpsConvolutionWindowTest() {};

void AlpsConvolutionWindowTest::SetUp() {};

void AlpsConvolutionWindowTest::TearDown() {};

TEST_F(AlpsConvolutionWindowTest, ByDefaultGetStatusIsTrue) {
  using Tensor = Alps::Tensor< int, 1 >;
  Alps::Window< Tensor > window_1( /* half window size */ {2},
				   /* padding */          {1},
				   /* striding */         {1});
  EXPECT_EQ(true,true);
}

//TEST_F(AlpsConvolutionWindowTest, ByDefaultBazFalseIsFalse) {
//    LoadDataSet foo(m_bar);
//    EXPECT_EQ(foo.baz(false), false);
//}
//
//TEST_F(AlpsConvolutionWindowTest, SometimesBazFalseIsTrue) {
//    LoadDataSet foo(m_bar);
//    // Have norf return true for once
//    EXPECT_CALL(m_bar,norf()).WillOnce(Return(true));
//    EXPECT_EQ(foo.baz(false), true);
//}

