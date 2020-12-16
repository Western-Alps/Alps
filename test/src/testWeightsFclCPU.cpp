#include "testWeightsFclCPU.h"
#include "AlpsLoadDataSet.h"
#include "AlpsWeightsFclCPU.h"
#include "AlpsTools.h"

//using ::testing::Return;

WeightsFclCPUTest::WeightsFclCPUTest() {
    // Have qux return true by default
    //ON_CALL(m_bar,qux()).WillByDefault(Return(true));
    // Have norf return false by default
    //ON_CALL(m_bar,norf()).WillByDefault(Return(false));
}

WeightsFclCPUTest::~WeightsFclCPUTest() {};

void WeightsFclCPUTest::SetUp() {};

void WeightsFclCPUTest::TearDown() {};
//
// Constructor
TEST_F(WeightsFclCPUTest, ByDefaultSubjectZero) {
  // Constructor of a subject
  Alps::WeightsFclCPU W( nullptr,
			 std::vector< int >(1, 5),
			 std::vector< int >(1, 10) );
  //
  //
  // EXPECT_EQ( W.get_subject_number(), 0) ;
  EXPECT_EQ( (*W.get_weight())(0,1), (*W.get_weight_transposed())(1,0) ) ;
}
//// Add modalities
//TEST_F(WeightsFclCPUTest, ByDefaultSubjectAddModalitiesTrue) {
//  // Constructor of a subject
//  Alps::SubjectCPU< 2 > Subj = Alps::SubjectCPU< 2 >( 0, 2);
//  // load modalities
//  Subj.add_modalities("../images/MNITS/000000-num5.png");
//  Subj.add_modalities("../images/MNITS/000000-num5.png");
//  //
//  //
//  EXPECT_EQ( Subj.check_modalities(), true) ;
//}

//TEST_F(WeightsFclCPUTest, ByDefaultBazFalseIsFalse) {
//    Subjects foo(m_bar);
//    EXPECT_EQ(foo.baz(false), false);
//}
//
//TEST_F(WeightsFclCPUTest, SometimesBazFalseIsTrue) {
//    Subjects foo(m_bar);
//    // Have norf return true for once
//    EXPECT_CALL(m_bar,norf()).WillOnce(Return(true));
//    EXPECT_EQ(foo.baz(false), true);
//}

