#include "testAlpsFullSamples.h"
#include "AlpsFullSamples.h"
#include "AlpsLoadDataSet.h"
#include "AlpsMountainDummy.h"

//using ::testing::Return;

FullSamplesTest::FullSamplesTest() {
    // Have qux return true by default
    //ON_CALL(m_bar,qux()).WillByDefault(Return(true));
    // Have norf return false by default
    //ON_CALL(m_bar,norf()).WillByDefault(Return(false));
}

FullSamplesTest::~FullSamplesTest() {};

void FullSamplesTest::SetUp() {};

void FullSamplesTest::TearDown() {};

TEST_F(FullSamplesTest, ByDefaultGetStatusIsTrue) {
  //
  // Load the dataset
  Alps::LoadDataSet::instance("data_set_GP.json");
  //
  // Test the Cross validation constructor 
  Alps::FullSamples< Alps::MountainDummy, /*Dim*/ 2 > cv;
  
  //
  //
  EXPECT_EQ( Alps::LoadDataSet::instance()->get_status(), true );
}

TEST_F(FullSamplesTest, ByDefaultGetTrainStatusIsTrue) {
  //
  // Load the dataset
  Alps::LoadDataSet::instance("data_set_GP.json");
  //
  // Test the Cross validation 
  Alps::FullSamples< Alps::MountainDummy, /*Dim*/ 2 > cv;
  cv.train();
  
  //
  //
  EXPECT_EQ( Alps::LoadDataSet::instance()->get_status(), true );
}
//TEST_F(FullSamplesTest, ByDefaultGetLoadITKIsTrue) {
//  Alps::LoadDataSet::instance("data_set_GP.json");
//  EXPECT_EQ(Alps::LoadDataSet::instance()->Load_ITK_images(), true);
//}

//TEST_F(FullSamplesTest, ByDefaultBazFalseIsFalse) {
//    FullSamples foo(m_bar);
//    EXPECT_EQ(foo.baz(false), false);
//}
//
//TEST_F(FullSamplesTest, SometimesBazFalseIsTrue) {
//    FullSamples foo(m_bar);
//    // Have norf return true for once
//    EXPECT_CALL(m_bar,norf()).WillOnce(Return(true));
//    EXPECT_EQ(foo.baz(false), true);
//}

