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
  // We have three convolutional layers. At the end of the three pulling image
  // we will have:
  std::vector< Image3DType::SizeType > fully_connected_input =
    MAC::Singleton::instance()->get_subjects()[0].get_modality_images_size();
  //
  for ( int mod = 0 ; mod < fully_connected_input.size() ; mod++ )
    for ( int dim = 0 ; dim < 3 ; dim++ )
      for ( int i = 0 ; i < 3 /* num conv layers*/ ; i++ )
	fully_connected_input[mod][dim] = static_cast< int >( fully_connected_input[mod][dim] / 2 );
  //
  for ( auto s : fully_connected_input )
    std::cout << s << std::endl;

  //
  // Anatomy
  mr_nn_.add( nn_0 );
  mr_nn_.add( nn_1 );
  mr_nn_.add( nn_2 );


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
