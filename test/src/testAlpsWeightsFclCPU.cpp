#include "testAlpsWeightsFclCPU.h"
#include "AlpsLoadDataSet.h"
#include "AlpsWeightsFclCPU.h"
#include "AlpsSubject.h"
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
TEST_F(WeightsFclCPUTest, ByDefaultWeigths) {
  // Constructor of a subject
  Alps::WeightsFclCPU W( nullptr,
			 std::vector< int >(1, 5),
			 std::vector< int >(1, 10) );
  //
  //
  EXPECT_EQ( 0, 0 ) ;
}
// Accessor
TEST_F(WeightsFclCPUTest, ByDefaultWeigthsGet) {
  // Constructor of weights
  Alps::WeightsFclCPU W( nullptr,
			 /*Current  layer*/ std::vector< int >(1, 2),
			 /*Previous layer*/ std::vector< int >(1, 3) );
  //
  auto weights   = W.get_weights()[0];
  auto weights_T = W.get_weights()[0].transpose();
  //
  //
  EXPECT_EQ( weights(1,2), weights_T(2,1) ) ;
}
// Accessor
TEST_F(WeightsFclCPUTest, ByDefaultWeigthsGet2) {
  // Constructor of weights
  Alps::WeightsFclCPU W( nullptr,
			 /*Current  layer*/ std::vector< int >(1, 2),
			 /*Previous layer*/ std::vector< int >(2, 3) );
  //
  auto weights   = W.get_weights()[0];
  auto weights_T = W.get_weights()[0].transpose();
  //
  //
  EXPECT_EQ( weights(1,2), weights_T(2,1) ) ;
}
// Activate
TEST_F(WeightsFclCPUTest, ByDefaultWeigthsActivate) {
  //
//ToDo  // Constructor of a subject
//ToDo  std::shared_ptr< Alps::Subject< 2 > > Subj = std::make_shared< Alps::Subject< 2 > >( Alps::Subject< 2 >( 0, 2) );
//ToDo  // load modalities
//ToDo  Subj->add_modalities("../images/MNITS/000000-num5.png");
//ToDo  Subj->add_modalities("../images/MNITS/000000-num5.png");
//ToDo  //
//ToDo  // Constructor of weights
//ToDo  Alps::WeightsFclCPU W( nullptr,
//ToDo			 /*Current  layer*/ std::vector< int >(1, 2),
//ToDo			 /*Previous layer*/ std::vector< int >(2, 3) );
//ToDo  //
//ToDo  //
//ToDo  W.activate( Subj->get_layer_modalities("__input_layer__") );
  //
  //
  EXPECT_EQ( 0, 0 ) ;
}
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

