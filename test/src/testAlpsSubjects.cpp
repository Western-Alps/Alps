#include "testAlpsSubjects.h"
#include "AlpsMountainDummy.h"
#include "AlpsLoadDataSet.h"
#include "AlpsSubjects.h"
#include "AlpsTools.h"

//using ::testing::Return;

SubjectsTest::SubjectsTest() {
    // Have qux return true by default
    //ON_CALL(m_bar,qux()).WillByDefault(Return(true));
    // Have norf return false by default
    //ON_CALL(m_bar,norf()).WillByDefault(Return(false));
}

SubjectsTest::~SubjectsTest() {};

void SubjectsTest::SetUp() {};

void SubjectsTest::TearDown() {};

TEST_F(SubjectsTest, ByDefaultSubjectsTrue) {
  // Load the dataset 
  Alps::LoadDataSet::instance("data_set_GP.json");
  // Creates the subjects container
  Alps::Subjects< /*Functions,*/ 2 > subjects( std::make_shared< Alps::MountainDummy >() );
  EXPECT_EQ(Alps::LoadDataSet::instance()->Load_ITK_images(), true);
}

//TEST_F(SubjectsTest, ByDefaultBazFalseIsFalse) {
//    Subjects foo(m_bar);
//    EXPECT_EQ(foo.baz(false), false);
//}
//
//TEST_F(SubjectsTest, SometimesBazFalseIsTrue) {
//    Subjects foo(m_bar);
//    // Have norf return true for once
//    EXPECT_CALL(m_bar,norf()).WillOnce(Return(true));
//    EXPECT_EQ(foo.baz(false), true);
//}

