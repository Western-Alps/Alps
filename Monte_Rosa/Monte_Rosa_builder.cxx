//
// CUDA
//
//
//
#include "MACException.h"
#include "NeuralNetwork.h"
#include "NeuralNetworkComposite.h"
#include "Monte_Rosa_builder.h"
//
//
//
MAC::Monte_Rosa_builder::Monte_Rosa_builder():
  MAC::NeuralNetwork::NeuralNetwork()
{
  //
  // Neural network anatomy
  //

   
  //
  // Convolutional layers
  //
  
  //
  // Layer 0
  int window_0[3] = {3,3,3};
  //
  std::shared_ptr< NeuralNetwork > nn_0 =
    std::make_shared< MAC::Convolutional_layer >( "layer_0", 0,
						  window_0 );

  //
  // Layer 1
  int window_1[3] = {5,5,5};
  std::shared_ptr< NeuralNetwork > nn_1 =
    std::make_shared< MAC::Convolutional_layer >( "layer_1", 1,
						  window_1 );
  
  //
  // Layer 2
  int window_2[3] = {11,11,11};
  std::shared_ptr< NeuralNetwork > nn_2 =
    std::make_shared< MAC::Convolutional_layer >( "layer_2", 2,
						  window_2 );

  //
  // Fully connected layers
  // First we have to know how many input we will have
  // Next we can build the layers
  //
  
  //
  // 1. Inputs of the fully connected layers neural network
  int number_of_inputs = 0;
  // We have three convolutional layers. At the end of the three pulling image
  // we will have:
  std::vector< Image3DType::SizeType > fully_connected_input =
    MAC::Singleton::instance()->get_subjects()[0].get_modality_images_size();
  //
  for ( int mod = 0 ; mod < static_cast< int >( fully_connected_input.size() ) ; mod++ )
    for ( int dim = 0 ; dim < 3 ; dim++ )
      for ( int i = 0 ; i < 3 /* num conv layers*/ ; i++ )
	fully_connected_input[mod][dim] = static_cast< int >( fully_connected_input[mod][dim] / 2 );
  //
  for ( auto mod : fully_connected_input )
    {
      int mod_num_weights = 1;
      for ( int dim = 0 ; dim < 3 ; dim++ )
	mod_num_weights *= mod[dim];
      //
      number_of_inputs += mod_num_weights;
    }
  std::cout << number_of_inputs << std::endl;
  //
  // 2. Create the fully connected layer
  // the +1 is for the bias weights
  const int num_fc_layers      = 6;
  int fc_layers[num_fc_layers] = {number_of_inputs+1,1000+1,500+1,100+1,50+1,3};
  //
  std::shared_ptr< NeuralNetwork > nn_3 =
    std::make_shared< MAC::FullyConnected_layer >( "layer_3", 3,
						   num_fc_layers,
						   fc_layers );
  

  //
  // Anatomy
  //
  
  mr_nn_.add( nn_0 );
  mr_nn_.add( nn_1 );
  mr_nn_.add( nn_2 );
  mr_nn_.add( nn_3 );

  //MAC::Singleton::instance()->get_subjects()[0].write_clone();
};
//
//
//
void
MAC::Monte_Rosa_builder::initialization()
{
  mr_nn_.initialization();
};
//
//
//
void
MAC::Monte_Rosa_builder::forward( Subject& Sub, const Weights& W )
{
  mr_nn_.forward( Sub, W );
};
