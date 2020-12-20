#include "testCVKFolds.h"
#include "AlpsCVKFolds.h"
#include "AlpsLoadDataSet.h"
#include "AlpsMountainDummy.h"

//using ::testing::Return;

CVKFoldsTest::CVKFoldsTest() {
    // Have qux return true by default
    //ON_CALL(m_bar,qux()).WillByDefault(Return(true));
    // Have norf return false by default
    //ON_CALL(m_bar,norf()).WillByDefault(Return(false));
}

CVKFoldsTest::~CVKFoldsTest() {};

void CVKFoldsTest::SetUp() {};

void CVKFoldsTest::TearDown() {};

TEST_F(CVKFoldsTest, ByDefaultGetStatusIsTrue) {
  //
  // Load the dataset
  Alps::LoadDataSet::instance("data_set_GP.json");
  //
  // Test the Cross validation constructor 
  Alps::CVKFolds< 3, Alps::MountainDummy > cv;
  
  //
  //
  EXPECT_EQ( Alps::LoadDataSet::instance()->get_status(), true );
}

TEST_F(CVKFoldsTest, ByDefaultGetTrainStatusIsTrue) {
  //
  // Load the dataset
  Alps::LoadDataSet::instance("data_set_GP.json");
  //
  // Test the Cross validation 
  Alps::CVKFolds< 3, Alps::MountainDummy > cv;
  cv.train();
  
  //
  //
  EXPECT_EQ( Alps::LoadDataSet::instance()->get_status(), true );
}
//TEST_F(CVKFoldsTest, ByDefaultGetLoadITKIsTrue) {
//  Alps::LoadDataSet::instance("data_set_GP.json");
//  EXPECT_EQ(Alps::LoadDataSet::instance()->Load_ITK_images(), true);
//}

//TEST_F(CVKFoldsTest, ByDefaultBazFalseIsFalse) {
//    CVKFolds foo(m_bar);
//    EXPECT_EQ(foo.baz(false), false);
//}
//
//TEST_F(CVKFoldsTest, SometimesBazFalseIsTrue) {
//    CVKFolds foo(m_bar);
//    // Have norf return true for once
//    EXPECT_CALL(m_bar,norf()).WillOnce(Return(true));
//    EXPECT_EQ(foo.baz(false), true);
//}

