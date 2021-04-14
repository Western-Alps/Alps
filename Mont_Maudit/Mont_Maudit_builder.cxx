// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
//
#include "MACException.h"
//#include "NeuralNetwork.h"
//#include "NeuralNetworkComposite.h"
#include "Mont_Maudit_builder.h"
#include "AlpsWindow.h"
#include "AlpsWeightsConvolution.h"
#include "AlpsSGD.h"
#include "AlpsActivations.h"
#include "AlpsCostFunction.h"
#include "AlpsConvolutionLayer.h"
#include "AlpsTransposedConvolutionLayer.h"
//
//
//
Alps::Mont_Maudit_builder::Mont_Maudit_builder()
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
  using Activation       = Alps::Activation_tanh< double >;
  using Kernel           = Alps::Window< double, 2 >;
  using Weights          = Alps::WeightsConvolution< double, Eigen::MatrixXd, Alps::Arch::CPU, Activation, Alps::SGD, 2 >;
  using LossFunction     = Alps::LeastSquarreEstimate< double >;
  using Convolutional    = Alps::ConvolutionLayer< Activation, Weights, Kernel, LossFunction, /*Dim*/ 2 >;
  using Deconvolutional  = Alps::TransposedConvolutionLayer< Activation, Weights, Kernel, LossFunction, /*Dim*/ 2 >;

  //
  // Convolutional layers
  //
  // layer 1
  // Window definition for the kernels
  std::vector< long int > h_window_1 = {1,2}; // size of the 1/2 window
  std::vector< long int > padding_1  = {2,3}; // padding
  std::vector< long int > striding_1 = {1,1}; // striding
  //
  std::shared_ptr< Kernel > window_1 = std::make_shared< Kernel >( 10, // number of kernels
								   h_window_1, padding_1, striding_1 );
  //
  std::shared_ptr< Alps::Layer > nn_1 =
    std::make_shared< Convolutional >( "layer_1",
				       window_1 );
  nn_1->add_layer( nullptr );   // connection with the previous layer. (nullptr) means input layer
  //
  // layer 2
  // Window definition for the kernels. This window is centered on one voxel.
  // If the padding and stridding is the same as the window 1, the output image
  // will have exactly the same dimension.
  std::vector< long int > h_window_2 = {0,0}; // size of the 1/2 window, when the value is 0, the kernel is the size of a pixel
  std::vector< long int > padding_2  = {2,3}; // padding
  std::vector< long int > striding_2 = {1,1}; // striding
  //
  std::shared_ptr< Kernel > window_2 = std::make_shared< Kernel >( 20, // number of kernels
								   h_window_2, padding_2, striding_2 );
  //
  std::shared_ptr< Alps::Layer > nn_2 =
    std::make_shared< Convolutional >( "layer_2",
				       window_2 );
  nn_2->add_layer( nullptr ); // connection with the previous layer. (nullptr) means input layer
  //
  // layer 3
  // Window definition for the kernels. This window is centered on one voxel.
  std::vector< long int > h_window_3 = {4,4}; // size of the 1/2 window
  std::vector< long int > padding_3  = {0,0}; // padding
  std::vector< long int > striding_3 = {1,1}; // striding
  //
  std::shared_ptr< Kernel > window_3 = std::make_shared< Kernel >( 10, // number of kernels
								   h_window_3, padding_3, striding_3 );
  //
  std::shared_ptr< Alps::Layer > nn_3 =
    std::make_shared< Convolutional >( "layer_3",
				       window_3 );
  nn_3->add_layer( nn_1 ); 
  nn_3->add_layer( nn_2 );
  //
  // layer 4
  // Window definition for the kernels
  std::vector< long int > h_window_4 = {4,4}; // size of the 1/2 window
  std::vector< long int > padding_4  = {0,0}; // padding
  std::vector< long int > striding_4 = {2,2}; // striding
  //
  std::shared_ptr< Kernel > window_4 = std::make_shared< Kernel >( 4, // number of kernels
								   h_window_4, padding_4, striding_4 );
  //
  std::shared_ptr< Alps::Layer > nn_4 =
    std::make_shared< Convolutional >( "layer_4",
				       window_4 );
  nn_4->add_layer( nn_3 );   // inputs and nn_2 are inputs of nn_1, then nn_1 is input of nn_4



  //
  // Deconvolutional layers
  //
  // layer 5
  //
  std::shared_ptr< Alps::Layer > nn_5 =
    std::make_shared< Deconvolutional >( "layer_5",
					  std::dynamic_pointer_cast< Convolutional >(nn_2) );
  nn_5->add_layer( nn_4 );
  //
  // layer 6
  //
  std::shared_ptr< Alps::Layer > nn_6 =
    std::make_shared< Deconvolutional >( "__output_layer__",
					 nullptr, false );
  nn_6->add_layer( nn_5 );   // inputs and nn_2 are inputs of nn_1, then nn_1 is input of nn_4
  //
  // layer 7
  //
  std::shared_ptr< Alps::Layer > nn_7 =
    std::make_shared< Deconvolutional >( "__output_layer__",
					 nullptr, false );
  nn_7->add_layer( nn_5 );   // inputs and nn_2 are inputs of nn_1, then nn_1 is input of nn_4
  //
  //
  // The output of nn_{6,7} should be combined togther as a linear combinaison in a non-linear
  // function, e.g. sigmoid, with one bias:
  // nn_8 = sigmoid( nn_6 + nn_7 + b). Then nn_8 should be directly compared to the target
  // in the cost function.

  
  
  /////////////
  // Anatomy //
  /////////////
  mr_nn_.add( nn_1 );
  mr_nn_.add( nn_2 );
  mr_nn_.add( nn_4 );
//  mr_nn_.add( nn_4 );
//  mr_nn_.add( nn_5 );
//  mr_nn_.add( nn_6 );
};
//
//
//
void
Alps::Mont_Maudit_builder::forward( std::shared_ptr< Alps::Climber > Sub )
{
  mr_nn_.forward( Sub );
};
//
//
//
void
Alps::Mont_Maudit_builder::backward( std::shared_ptr< Alps::Climber > Sub )
{
  mr_nn_.backward( Sub );
};
//
//
//
void
Alps::Mont_Maudit_builder::weight_update( std::shared_ptr< Alps::Climber > Sub )
{
  mr_nn_.weight_update( Sub );
};
//
//
//
void
Alps::Mont_Maudit_builder::notify()
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
