//
// CUDA
//
//
//
#include "MACException.h"
#include "NeuralNetwork.h"
#include "NeuralNetworkComposite.h"
#include "Gran_Paradiso_builder.h"
#include "AlpsWeightsFclCPU.h"
#include "Activations.h"
//
//
//
MAC::Gran_Paradiso_builder::Gran_Paradiso_builder()
{
  //
  // Neural network anatomy
  //
  using weights        = Alps::WeightsFclCPU;
  using FullyConnected = Alps::FullyConnectedLayer< Activation_tanh, weights, 2 >;
   
  //
  // Fully connected layers
  // The *bias* is not included
  std::shared_ptr< Alps::Layer > nn_1 =
    std::make_shared< FullyConnected >( "layer_1",
					std::vector<int>( 1, 10 ), // 1 layer of 10 elements
					nullptr );
  //
  std::shared_ptr< Alps::Layer > nn_2 =
    std::make_shared< FullyConnected >( "layer_2",
					std::vector<int>( 1, 5 ), // 1 layer of 5 elements
					nn_1 );
  nn_1->set_next_layer( nn_2 );
  //
  std::shared_ptr< Alps::Layer > nn_3 =
    std::make_shared< FullyConnected >( "output",
					std::vector<int>( 1, 3 ), // 1 layer of 3 elements
					nn_2 );
  nn_2->set_next_layer( nn_3 );
  

  //
  // Anatomy
  //
  
  mr_nn_.add( nn_1 );
  mr_nn_.add( nn_2 );
  mr_nn_.add( nn_3 );

  //MAC::Singleton::instance()->get_subjects()[0].write_clone();
};
//
//
//
//void
//MAC::Gran_Paradiso_builder::initialization()
//{
//  mr_nn_.initialization();
//};
//
//
//
void
MAC::Gran_Paradiso_builder::forward( std::shared_ptr< Alps::Climber > Sub )
{
  mr_nn_.forward( Sub );
};
//
//
//
void
MAC::Gran_Paradiso_builder::backward()
{
  mr_nn_.backward();
};
