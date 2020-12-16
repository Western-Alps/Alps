#include "testImage.h"
#include "AlpsImage.h"
#include "AlpsLoadDataSet.h"
//#include "AlpsTools.h"

//using ::testing::Return;

ImageTest::ImageTest() {
    // Have qux return true by default
    //ON_CALL(m_bar,qux()).WillByDefault(Return(true));
    // Have norf return false by default
    //ON_CALL(m_bar,norf()).WillByDefault(Return(false));
}

ImageTest::~ImageTest() {};

void ImageTest::SetUp() {};

void ImageTest::TearDown() {};
//
// Constructor
TEST_F(ImageTest, ByDefaultSubjectZero) {
  // Constructor of a subject
  Alps::Image< 2 > Subj = Alps::Image< 2 >( nullptr,
					    std::vector< size_t >(2,5) );
  //
  //
  EXPECT_EQ( 0, 0) ;
}
//// Add modalities
//TEST_F(ImageTest, ByDefaultSubjectAddModalitiesTrue) {
//  // Constructor of a subject
//  Alps::Image< 2 > Subj = Alps::Image< 2 >( 0, 2);
//  // load modalities
//  Subj.add_modalities("../images/MNITS/000000-num5.png");
//  Subj.add_modalities("../images/MNITS/000000-num5.png");
//  //
//  //
//  EXPECT_EQ( Subj.check_modalities(), true) ;
//}

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

