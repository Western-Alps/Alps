#include "testAlpsSubject.h"
#include "AlpsLoadDataSet.h"
#include "AlpsSubjects.h"
#include "AlpsTools.h"
#include "AlpsLayerTensors.h"

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
  Alps::Subject< 2 > Subj = Alps::Subject< 2 >( 0, 1 );
  //
  //
  EXPECT_EQ( Subj.get_subject_number(), 0) ;
}
// Add modalities
TEST_F(SubjectTest, ByDefaultSubjectAddModalitiesTrue) {
  // Constructor of a subject
  Alps::Subject< 2 > Subj = Alps::Subject< 2 >( 0, 2 );
  // load modalities
  Subj.add_modalities("../images/MNITS/000000-num5.png");
  Subj.add_modalities("../images/MNITS/000000-num5.png");

  //
  //
  //std::cout << "Size: " << Subj.get_layer_size( "__input_layer__" )[0] << std::endl;
  //std::cout << "position 547: " << Subj.get_layer( "__input_layer__" )[0][Alps::TensorOrder1::ACTIVATION][547] << std::endl;
  //std::cout << "position 784+547: " << Subj.get_layer( "__input_layer__" )[0][Alps::TensorOrder1::ACTIVATION][784+547] << std::endl;
  //
  //  std::size_t size = Subj.get_layer_size( "__input_layer__" )[0];
  //  for (std::size_t i = 0 ; i < size ; i++ )
  //    std::cout << "position "<< i <<": " << Subj.get_layer( "__input_layer__" )[0][Alps::TensorOrder1::ACTIVATION][i] << std::endl;

  
  //
  //
  EXPECT_EQ( Subj.get_layer( "__input_layer__" )[0][Alps::TensorOrder1::ACTIVATION][547],
	     Subj.get_layer( "__input_layer__" )[0][Alps::TensorOrder1::ACTIVATION][784+547] );
}
// Add modalities
TEST_F(SubjectTest, ByDefaultSubjectAddTarget) {
  // Constructor of a subject
  Alps::Subject< 2 > Subj = Alps::Subject< 2 >( 0, 1);
  // load modalities
  Subj.add_modalities("../images/MNITS/000000-num5.png");
  // Add targetted value
  Subj.add_target( 5., 10.);
  //
  //
  EXPECT_EQ( (Subj.get_target().get_tensor())[5], 1. ) ;
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

