#include "testAlpsTensor.h"
#include "AlpsTensor.h"


//using ::testing::Return;

AlpsTensorTest::AlpsTensorTest() {
    // Have qux return true by default
    //ON_CALL(m_bar,qux()).WillByDefault(Return(true));
    // Have norf return false by default
    //ON_CALL(m_bar,norf()).WillByDefault(Return(false));
}

AlpsTensorTest::~AlpsTensorTest() {};

void AlpsTensorTest::SetUp() {};

void AlpsTensorTest::TearDown() {};

///////////////
// ACCESSORS //
///////////////
////
//// Access dimention of the space
//// Dimension 0
//TEST_F(AlpsTensorTest, Constructor)
//{
//  EXPECT_ANY_THROW( Alps::Tensor< int, 0 >() );
//}
// Dimension 1
TEST_F(AlpsTensorTest, GetDim1)
{
  Alps::Tensor< int, 1 > tensor_1;
  ASSERT_EQ( tensor_1.get_dimension(), 1 );
}
// Dimension 2
TEST_F(AlpsTensorTest, GetDim2)
{
  Alps::Tensor< int, 2 > tensor_2;
  ASSERT_EQ( tensor_2.get_dimension(), 2 );
}
// Dimension 3
TEST_F(AlpsTensorTest, GetDim3)
{
  Alps::Tensor< int, 3 > tensor_3;
  ASSERT_EQ( tensor_3.get_dimension(), 3 );
}
// Dimension 4
TEST_F(AlpsTensorTest, GetDim4)
{
  Alps::Tensor< int, 4 > tensor_4;
  ASSERT_EQ( tensor_4.get_dimension(), 4 );
}
//
// Access the size of a dimension
TEST_F(AlpsTensorTest, GetTensor1Size0)
{
  Alps::Tensor< int, 1 > tensor_1;
  ASSERT_EQ( tensor_1.get_size( 0 ), 0 );
}
//TEST_F(AlpsTensorTest, SometimesBazFalseIsTrue) {
//    LoadDataSet foo(m_bar);
//    // Have norf return true for once
//    EXPECT_CALL(m_bar,norf()).WillOnce(Return(true));
//    EXPECT_EQ(foo.baz(false), true);
//}

