// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
//
#include "MACException.h"
//#include "NeuralNetwork.h"
//#include "NeuralNetworkComposite.h"
#include "Monte_Rosa_builder.h"
#include "AlpsWeightsFcl.h"
#include "AlpsSGD.h"
#include "AlpsActivations.h"
#include "AlpsCostFunction.h"
//
//
//
Alps::Monte_Rosa_builder::Monte_Rosa_builder()
{
  //
  // 
  energy_.push_back( 1.e+6 );
  
  //
  // Create a unique id for the layer
  std::random_device                   rd;
  std::mt19937                         generator( rd() );
  std::uniform_int_distribution< int > distribution( 0, 1UL << 16 );
  //
  layer_id_ = distribution( generator );
  

  ////////////////////////////
  // Neural network anatomy //
  ////////////////////////////
  using Activation     = Alps::Activation_tanh< double >;
  using Weights        = Alps::WeightsFcl< double, Eigen::MatrixXd, Alps::Arch::CPU, Activation, Alps::SGD >;
  using LossFunction   = Alps::LeastSquarreEstimate< double >;
//  //
//  // Neural network anatomy
//  //
//  using Convolution    = Alps::Convolutional_layer< Activation_tanh >;
//  using FullyConnected = Alps::FullyConnected_layer< Activation_tanh >;
//   
//  //
//  // Convolutional layers
//  //
//  int downsize_factor = 2;
//  
//  //
//  // Layer 0
//  // {s,x,y,z}
//  // s: num of feature maps we want to create
//  // (x,y,z) sive of the receiving window
//  int window_0[4] = {5 /*s*/, 3 /*x*/,3 /*y*/,3 /*z*/};
//  //
//  std::shared_ptr< NeuralNetwork > nn_0 =
//    std::make_shared< Convolution >( "layer_0", 0,
//				     downsize_factor,
//				     window_0 );
//
//  //
//  // Layer 1
//  int window_1[4] = {10,5,5,5};
//  std::shared_ptr< NeuralNetwork > nn_1 =
//    std::make_shared< Convolution >( "layer_1", 1,
//				     downsize_factor,
//				     window_1 );
//  
//  //
//  // Layer 2
//  int window_2[4] = {20,3,3,3};
//  std::shared_ptr< NeuralNetwork > nn_2 =
//    std::make_shared< Convolution >( "layer_2", 2,
//				     downsize_factor,
//				     window_2 );
//
//  //
//  // Fully connected layers
//  // the +1 is for the bias weights
//  const int num_fc_layers = 6;
//  // "-1" for the input layer, when we don't know yet how many inputs we will have
//  // The bias is not included
//  int fc_layers[num_fc_layers] = { -1, 1000, 500, 100, 50, 3 };
//  //
//  std::shared_ptr< NeuralNetwork > nn_3 =
//    std::make_shared< FullyConnected >( "layer_3", 3,
//					num_fc_layers,
//					fc_layers );
//  
//
//  //
//  // Anatomy
//  //
//  
//  mr_nn_.add( nn_0 );
//  mr_nn_.add( nn_1 );
//  mr_nn_.add( nn_2 );
//  mr_nn_.add( nn_3 );
//
//  //Alps::Singleton::instance()->get_subjects()[0].write_clone();
};
//
//
//
void
Alps::Monte_Rosa_builder::forward( std::shared_ptr< Alps::Climber > Sub  )
{
  mr_nn_.forward( Sub );
};
//
//
//
void
Alps::Monte_Rosa_builder::backward( std::shared_ptr< Alps::Climber > Sub )
{
  mr_nn_.backward( Sub );
};
//
//
//
void
Alps::Monte_Rosa_builder::weight_update( std::shared_ptr< Alps::Climber > Sub )
{
  mr_nn_.weight_update( Sub );
};
//
//
//
void
Alps::Monte_Rosa_builder::notify()
{
  //
  // add energy for all subject in the epoque
  std::cout << "num subjects: " << energy_subject_.size() << std::endl;
  double cost = 0.;
  for ( auto e : energy_subject_ )
    cost += e;
  // reset the energy for the next epoque
  energy_subject_.clear();
  // record the energy for the epoque
  energy_.push_back( cost );
  //
  std::cout << "epoque: " << energy_.size() << std::endl;
};
