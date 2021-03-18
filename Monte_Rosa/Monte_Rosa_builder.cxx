// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
//
#include "MACException.h"
//#include "NeuralNetwork.h"
//#include "NeuralNetworkComposite.h"
#include "Monte_Rosa_builder.h"
#include "AlpsWindow.h"
#include "AlpsWeightsConvolution.h"
#include "AlpsWeightsFcl.h"
#include "AlpsSGD.h"
#include "AlpsActivations.h"
#include "AlpsCostFunction.h"
#include "AlpsFullyConnectedLayer.h"
#include "AlpsConvolutionLayer.h"
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
  using WeightsConv    = Alps::WeightsConvolution< double, Eigen::MatrixXd, Alps::Arch::CPU, Activation, Alps::SGD, 2 >;
  using Kernel         = Alps::Window< double, 2 >;
  using LossFunction   = Alps::LeastSquarreEstimate< double >;
  using FullyConnected = Alps::FullyConnectedLayer< Activation, Weights, LossFunction, /*Dim*/ 2 >;
  using Convolutional  = Alps::ConvolutionLayer< Activation, WeightsConv, Kernel, LossFunction, /*Dim*/ 2 >;

  //
  // Convolutional layers
  //
  // layer 1
  // Window definition for the kernels
  std::vector< long int > h_window_1 = {2,3}; // size of the 1/2 window
  std::vector< long int > padding_1  = {2,3}; // padding
  std::vector< long int > striding_1 = {1,1}; // striding
  //
  std::shared_ptr< Kernel > window_1 = std::make_shared< Kernel >( 10, // number of kernels
								   h_window_1, padding_1, striding_1 );
  //
  //
  std::shared_ptr< Alps::Layer > nn_1 =
    std::make_shared< Convolutional >( "layer_1",
				       window_1 );
  nn_1->add_layer( nullptr );   // connection with the previous layer. (nullptr) means input layer
  //
  // layer 2
  // Window definition for the kernels. This second window is a simple vovel size convolution window.
  std::vector< long int > h_window_2 = {0,0}; // size of the 1/2 window
  std::vector< long int > padding_2  = {0,0}; // padding
  std::vector< long int > striding_2 = {1,1}; // striding
  //
  std::shared_ptr< Kernel > window_2 = std::make_shared< Kernel >( 10, // number of kernels
								   h_window_2, padding_2, striding_2 );
  //
  std::shared_ptr< Alps::Layer > nn_2 =
    std::make_shared< Convolutional >( "layer_2",
				       window_2 );
  nn_2->add_layer( nullptr ); // connection with the previous layer. (nullptr) means input layer
  nn_1->add_layer( nn_2 );   // We should be able to do that since nn_2 has the same dimensions as the input
  //
  // layer 3
  // Window definition for the kernels
  std::vector< long int > h_window_3 = {5,5}; // size of the 1/2 window
  std::vector< long int > padding_3  = {0,0}; // padding
  std::vector< long int > striding_3 = {2,2}; // striding
  //
  std::shared_ptr< Kernel > window_3 = std::make_shared< Kernel >( 5, // number of kernels
								   h_window_3, padding_3, striding_3 );
  //
  std::shared_ptr< Alps::Layer > nn_3 =
    std::make_shared< Convolutional >( "layer_3",
				       window_3 );
  nn_3->add_layer( nn_1 );   // inputs and nn_2 are inputs of nn_1, then nn_1 is input of nn_3


  //
  // Fully connected layers
  //
  // The *bias* is not included
  std::shared_ptr< Alps::Layer > nn_4 =
    std::make_shared< FullyConnected >( "layer_4",
					std::vector<std::size_t>( 1, 50 ) );// 1 layer of 10 elements 
  nn_4->add_layer( nn_3 );   // connection one-to-n with the previous layer. (nullptr) means input layer
  //
  std::shared_ptr< Alps::Layer > nn_5 =
    std::make_shared< FullyConnected >( "__output_layer__", // __output_layer__ signal it is the last one
					std::vector<std::size_t>( 1, 10 ) );// 1 layer of 3 elements
  nn_5->add_layer( nn_3 );   // connection one-to-n with the previous layer. (nullptr) means input layer
  nn_5->add_layer( nn_4 );   // connection one-to-n with the previous layer

  
  /////////////
  // Anatomy //
  /////////////
  mr_nn_.add( nn_1 );
  mr_nn_.add( nn_2 );
  mr_nn_.add( nn_3 );
  mr_nn_.add( nn_4 );
  mr_nn_.add( nn_5 );
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
