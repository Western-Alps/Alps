#include "testAlpsSubject.h"
#include "AlpsLoadDataSet.h"
#include "AlpsSubjects.h"
#include "AlpsTools.h"

//using ::testing::Return;

SubjectTest::SubjectTest() {
    // Have qux return true by default
    //ON_CALL(m_bar,qux()).WillByDefault(Return(true));
    // Have norf return false by default
    //ON_CALL(m_bar,norf()).WillByDefault(Return(false));
}

SubjectTest::~SubjectTest() {};

void SubjectTest::SetUp() {};

void SubjectTest::TearDown() {};
//
// Constructor
TEST_F(SubjectTest, ByDefaultSubjectZero) {
  // Constructor of a subject
  Alps::Subject< 2 > Subj = Alps::Subject< 2 >( 0, 1);
  //
  //
  EXPECT_EQ( Subj.get_subject_number(), 0) ;
}
// Add modalities
TEST_F(SubjectTest, ByDefaultSubjectAddModalitiesTrue) {
  // Constructor of a subject
  Alps::Subject< 2 > Subj = Alps::Subject< 2 >( 0, 2);
  // load modalities
  Subj.add_modalities("../images/MNITS/000000-num5.png");
  Subj.add_modalities("../images/MNITS/000000-num5.png");
  //
  //
  EXPECT_EQ( Subj.check_modalities("__input_layer__"), true) ;
}

//TEST_F(SubjectTest, ByDefaultBazFalseIsFalse) {
//    Subjects foo(m_bar);
//    EXPECT_EQ(foo.baz(false), false);
//}
//
//TEST_F(SubjectTest, SometimesBazFalseIsTrue) {
//    Subjects foo(m_bar);
//    // Have norf return true for once
//    EXPECT_CALL(m_bar,norf()).WillOnce(Return(true));
//    EXPECT_EQ(foo.baz(false), true);
//}

