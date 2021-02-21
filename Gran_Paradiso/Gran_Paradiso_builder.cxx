// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
//
#include "MACException.h"
#include "NeuralNetwork.h"
#include "NeuralNetworkComposite.h"
#include "Gran_Paradiso_builder.h"
#include "AlpsWeightsFcl.h"
#include "AlpsSGD.h"
#include "AlpsActivations.h"
#include "AlpsCostFunction.h"
//
//
//
Alps::Gran_Paradiso_builder::Gran_Paradiso_builder()
{
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
  using Solver         = Alps::SGD;
  using Weights        = Alps::WeightsFcl< double, Eigen::MatrixXd, Alps::Arch::CPU, Activation, Alps::SGD >;
  using LossFunction   = Alps::LeastSquarreEstimate< double >;
  using FullyConnected = Alps::FullyConnectedLayer< Activation, Weights, LossFunction, /*Dim*/ 2 >;

  //
  // Fully connected layers
  // The *bias* is not included
  std::shared_ptr< Alps::Layer > nn_1 =
    std::make_shared< FullyConnected >( "layer_1",
					std::vector<std::size_t>( 1, 50 ) );// 1 layer of 10 elements 
  nn_1->add_layer( nullptr );   // connection one-to-n with the previous layer. (nullptr) means input layer
  //
  std::shared_ptr< Alps::Layer > nn_2 =
    std::make_shared< FullyConnected >( "layer_2",
					std::vector<std::size_t>( 1, 30 ) ); // 1 layer of 5 elements
  nn_1->set_next_layer( nn_2 ); // connection one-to-one with the next layer
  nn_2->add_layer( nn_1 );      // connection one-to-n with the previous layer
  //
  std::shared_ptr< Alps::Layer > nn_3 =
    std::make_shared< FullyConnected >( "__output_layer__", // __output_layer__ signal it is the last one
					std::vector<std::size_t>( 1, 10 ) );// 1 layer of 3 elements
  nn_2->set_next_layer( nn_3 ); // connection one-to-one with the next layer
  nn_3->add_layer( nullptr );   // connection one-to-n with the previous layer. (nullptr) means input layer
  nn_3->add_layer( nn_2 );      // connection one-to-n with the previous layer

  /////////////
  // Anatomy //
  /////////////
  mr_nn_.add( nn_1 );
  mr_nn_.add( nn_2 );
  mr_nn_.add( nn_3 );
};
//
//
//
void
Alps::Gran_Paradiso_builder::forward( std::shared_ptr< Alps::Climber > Sub )
{
  mr_nn_.forward( Sub );
  energy_ += Sub->get_energy();
  std::cout << "Gran Paradiso: " << energy_ << std::endl;
};
//
//
//
void
Alps::Gran_Paradiso_builder::backward( std::shared_ptr< Alps::Climber > Sub )
{
  mr_nn_.backward( Sub );
};
//
//
//
void
Alps::Gran_Paradiso_builder::notify()
{
  energy_ = 0.;
};
