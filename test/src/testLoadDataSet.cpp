#include "testLoadDataSet.h"
#include "AlpsLoadDataSet.h"

//using ::testing::Return;

LoadDataSetTest::LoadDataSetTest() {
    // Have qux return true by default
    //ON_CALL(m_bar,qux()).WillByDefault(Return(true));
    // Have norf return false by default
    //ON_CALL(m_bar,norf()).WillByDefault(Return(false));
}

LoadDataSetTest::~LoadDataSetTest() {};

void LoadDataSetTest::SetUp() {};

void LoadDataSetTest::TearDown() {};

TEST_F(LoadDataSetTest, ByDefaultGetStatusIsTrue) {
  MAC::LoadDataSet::instance("neural_net.json");
  EXPECT_EQ(MAC::LoadDataSet::instance()->get_status(), true);
}

//TEST_F(LoadDataSetTest, ByDefaultBazFalseIsFalse) {
//    LoadDataSet foo(m_bar);
//    EXPECT_EQ(foo.baz(false), false);
//}
//
//TEST_F(LoadDataSetTest, SometimesBazFalseIsTrue) {
//    LoadDataSet foo(m_bar);
//    // Have norf return true for once
//    EXPECT_CALL(m_bar,norf()).WillOnce(Return(true));
//    EXPECT_EQ(foo.baz(false), true);
//}

