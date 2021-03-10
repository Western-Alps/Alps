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
  using Activation     = Alps::Activation_tanh< double >;
  using Kernel         = Alps::Window< double, 2 >;
  using Weights        = Alps::WeightsConvolution< double, Eigen::MatrixXd, Alps::Arch::CPU, Activation, Alps::SGD, 2 >;
  
  using LossFunction   = Alps::LeastSquarreEstimate< double >;
  using Convolutional  = Alps::ConvolutionLayer< Activation, Weights, LossFunction, /*Dim*/ 2 >;

  //
  // Convolutional layers
  //
  // layer 1
  // Window definition for the kernels
  Kernel window_1( 10,    // number of kernels
		   {2,3}, // size of the 1/2 window
		   {2,3}, // padding
		   {1,1}  /* striding */ );
//  //
//  std::shared_ptr< Alps::Layer > nn_1 =
//    std::make_shared< Convolutional >( "layer_1",
//				       window_1 );
//  nn_1->add_layer( nullptr );   // connection with the previous layer. (nullptr) means input layer
//  //
//  // layer 2
//  // Window definition for the kernels
//  Kernel window_2( 10, {0,0}, {0,0}, {1,1} );
//  //
//  std::shared_ptr< Alps::Layer > nn_2 =
//    std::make_shared< Convolutional >( "layer_2",
//				       window_2 );
//  nn_2->add_layer( nullptr ); // connection with the previous layer. (nullptr) means input layer
//  nn_1->add_layer( nn_2 );   // We should be able to do that since nn_2 has the same dimensions as the input
//  //
//  // layer 3
//  // Window definition for the kernels
//  Kernel window_3( 5, {5,5}, {0,0}, {2,2} );
//  //
//  std::shared_ptr< Alps::Layer > nn_3 =
//    std::make_shared< Convolutional >( "layer_3",
//				       window_3 );
//  nn_3->add_layer( nn_1 );   // inputs and nn_2 are inputs of nn_1, then nn_1 is input of nn_3
//
//
//
//  //
//  // Deconvolutional layers
//  //
//  // layer 4
//  //
//  std::shared_ptr< Alps::Layer > nn_4 =
//    std::make_shared< Deconvolutional >( "layer_4",
//					 nn_2 );
//  nn_4->add_layer( nn_3 );
//  //
//  // layer 5
//  //
//  std::shared_ptr< Alps::Layer > nn_5 =
//    std::make_shared< Deconvolutional >( "layer_5",
//					 nn_1 );
//  nn_5->add_layer( nn_4 );
//  //
//  // layer 6
//  //
//  std::shared_ptr< Alps::Layer > nn_6 =
//    std::make_shared< Deconvolutional >( "__output_layer__",
//					 nullptr );
//  nn_6->add_layer( nn_4 );   // inputs and nn_2 are inputs of nn_1, then nn_1 is input of nn_3
//  nn_6->add_layer( nn_5 );   // inputs and nn_2 are inputs of nn_1, then nn_1 is input of nn_3
//
//  
//  /////////////
//  // Anatomy //
//  /////////////
//  mr_nn_.add( nn_1 );
//  mr_nn_.add( nn_2 );
//  mr_nn_.add( nn_3 );
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
