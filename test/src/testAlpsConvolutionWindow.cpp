#include "testAlpsConvolutionWindow.h"
#include "AlpsTensor.h"
#include "AlpsWindow.h"

//using ::testing::Return;

ConvolutionWindowTest::ConvolutionWindowTest() {
    // Have qux return true by default
    //ON_CALL(m_bar,qux()).WillByDefault(Return(true));
    // Have norf return false by default
    //ON_CALL(m_bar,norf()).WillByDefault(Return(false));
}

ConvolutionWindowTest::~ConvolutionWindowTest() {};

void ConvolutionWindowTest::SetUp() {};

void ConvolutionWindowTest::TearDown() {};

TEST_F(ConvolutionWindowTest, ByDefaultGetStatusIsTrue) {
//ToDo  using Tensor = Alps::Tensor< int, 1 >;
//ToDo  Alps::Window< Tensor > window_1( /* half window size */ {2},
//ToDo				   /* padding */          {1},
//ToDo				   /* striding */         {1});
  EXPECT_EQ(true,true);
}

//TEST_F(ConvolutionWindowTest, ByDefaultBazFalseIsFalse) {
//    LoadDataSet foo(m_bar);
//    EXPECT_EQ(foo.baz(false), false);
//}
//
//TEST_F(ConvolutionWindowTest, SometimesBazFalseIsTrue) {
//    LoadDataSet foo(m_bar);
//    // Have norf return true for once
//    EXPECT_CALL(m_bar,norf()).WillOnce(Return(true));
//    EXPECT_EQ(foo.baz(false), true);
//}

