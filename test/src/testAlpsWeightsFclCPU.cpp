#include "testAlpsWeightsFclCPU.h"
#include "AlpsLoadDataSet.h"
#include "AlpsWeightsFcl.h"
#include "AlpsActivations.h"
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
  using Activation = Alps::Activation_tanh< double >;
  Alps::WeightsFcl< double, Eigen::MatrixXd, Alps::Arch::CPU, Activation > W( nullptr,
									      std::vector< std::size_t >(1, 5),
									      std::vector< std::size_t >(1, 10) );
  //
  //
  EXPECT_EQ( 0, 0 ) ;
}
// Accessor
TEST_F(WeightsFclCPUTest, ByDefaultWeigthsGet) {
  // Constructor of weights
  using Activation = Alps::Activation_tanh< double >;
  Alps::WeightsFcl< double, Eigen::MatrixXd, Alps::Arch::CPU, Activation > W( nullptr,
									      /*Current  layer*/ std::vector< std::size_t >(1, 2),
									      /*Previous layer*/ std::vector< std::size_t >(1, 3) );
  //
  auto weights   = *(W.get_tensor().get());
  auto weights_T = weights.transpose();
  //
  //
  EXPECT_EQ( weights(1,2), weights_T(2,1) ) ;
}
// Accessor
TEST_F(WeightsFclCPUTest, ByDefaultWeigthsGet2) {
  // Constructor of weights
  using Activation = Alps::Activation_tanh< double >;
  Alps::WeightsFcl< double, Eigen::MatrixXd, Alps::Arch::CPU, Activation > W( nullptr,
									      /*Current  layer*/ std::vector< std::size_t >(1, 2),
									      /*Previous layer*/ std::vector< std::size_t >(2, 3) );
  //
  auto weights   = *(W.get_tensor().get());
  auto weights_T = weights.transpose();
  //
  //
  EXPECT_EQ( weights(1,2), weights_T(2,1) ) ;
}
// Activate
TEST_F(WeightsFclCPUTest, ByDefaultWeigthsActivate) {
  //
  std::vector< std::size_t >
    ss(2),
    s1(1,3),
    s2(1,4);
  ss[0] = 3;
  ss[1] = 4;
  //
  std::shared_ptr< double >
    t1( new  double[3],
	std::default_delete< double[] >() ),
    t2( new  double[4],
	std::default_delete< double[] >() );
  //
  t1.get()[0] = -0.2;
  t1.get()[1] =  0.3;
  t1.get()[2] = -0.4;
  //
  t2.get()[0] =  0.4;
  t2.get()[1] = -0.3;
  t2.get()[2] =  0.2;
  t2.get()[3] = -0.1;
  //
  Alps::LayerTensors< double, 2 >
    lt1( s1, std::make_tuple(t1, nullptr, nullptr) ),
    lt2( s2, std::make_tuple(t2, nullptr, nullptr) );
  //
  using Activation = Alps::Activation_tanh< double >;
  Alps::WeightsFcl< double, Eigen::MatrixXd, Alps::Arch::CPU, Activation > W( nullptr,
									      /*Current  layer*/ std::vector< std::size_t >(1, 2),
									      /*Previous layer*/ ss );
  //
  auto weights = *( W.get_tensor().get() );
  Eigen::MatrixXd V( 8, 1);
  V << 1.,-0.2,0.3,-0.4,0.4,-0.3,0.2,-0.1;
  //
  auto a = weights * V;
  Activation func;
  std::cout << "Activation: \n" << a << std::endl;
  std::cout << "z(0,0): \n" << func.f( a(0,0) ) << std::endl;
  
  //
  //
  std::vector< Alps::LayerTensors< double, 2 > > prev_layer_tensors;
  prev_layer_tensors.push_back( lt1 );
  prev_layer_tensors.push_back( lt2 );
  //
  auto activ_res = W.activate( prev_layer_tensors );

  std::cout << "rest: " << (std::get< 0 >(activ_res).get())[0] << std::endl;
  

  //
  //
  EXPECT_EQ( func.f( a(0,0) ), (std::get< 0 >(activ_res).get())[0] ) ;
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

