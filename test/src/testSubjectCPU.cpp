#include "testSubjects.h"
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
//
// Constructor
TEST_F(SubjectsTest, ByDefaultSubjectZero) {
  // Constructor of a subject
  Alps::SubjectCPU< 2 > Subj = Alps::SubjectCPU< 2 >( 0, 1);
  //
  //
  EXPECT_EQ( Subj.get_subject_number(), 0) ;
}
// Add modalities
TEST_F(SubjectsTest, ByDefaultSubjectAddModalitiesTrue) {
  // Constructor of a subject
  Alps::SubjectCPU< 2 > Subj = Alps::SubjectCPU< 2 >( 0, 2);
  // load modalities
  Subj.add_modalities("../images/MNITS/000000-num5.png");
  Subj.add_modalities("../images/MNITS/000000-num5.png");
  //
  //
  EXPECT_EQ( Subj.check_modalities(), true) ;
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

