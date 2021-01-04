#include "MACException.h"
#include "NeuralNetwork.h"
#include "NeuralNetworkComposite.h"
#include "Gran_Paradiso_builder.h"
#include "AlpsWeightsFclCPU.h"
#include "AlpsActivations.h"
//
//
//
Alps::Gran_Paradiso_builder::Gran_Paradiso_builder()
{
  //
  // Neural network anatomy
  //
  using Weights        = Alps::WeightsFclCPU;
  using Activation     = Alps::Activation_tanh;
  using FullyConnected = Alps::FullyConnectedLayer< Activation, Weights, /*Dimension*/ 2 >;

  //
  // Fully connected layers
  // The *bias* is not included
  std::shared_ptr< Alps::Layer > nn_1 =
    std::make_shared< FullyConnected >( "layer_1",
					std::vector<int>( 1, 10 ) );// 1 layer of 10 elements 
  nn_1->add_layer( nullptr );   // connection one-to-n with the previous layer. (nullptr) means input layer
  //
  std::shared_ptr< Alps::Layer > nn_2 =
    std::make_shared< FullyConnected >( "layer_2",
					std::vector<int>( 1, 5 ) ); // 1 layer of 5 elements
  nn_1->set_next_layer( nn_2 ); // connection one-to-one with the next layer
  nn_2->add_layer( nn_1 );      // connection one-to-n with the previous layer
  //
  std::shared_ptr< Alps::Layer > nn_3 =
    std::make_shared< FullyConnected >( "output",
					std::vector<int>( 1, 3 ) );// 1 layer of 3 elements
  nn_2->set_next_layer( nn_3 ); // connection one-to-one with the next layer
  nn_3->add_layer( nullptr );   // connection one-to-n with the previous layer. (nullptr) means input layer
  nn_3->add_layer( nn_2 );      // connection one-to-n with the previous layer

  //
  // Anatomy
  //
  
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
};
//
//
//
void
Alps::Gran_Paradiso_builder::backward()
{
  mr_nn_.backward();
};
