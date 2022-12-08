// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
//
#include "MACException.h"
#include "Mont_Maudit_builder.h"
#include "AlpsWindow.h"
#include "AlpsWeightsConvolution.h"
#include "AlpsWeightsTransposedConvolution.h"
#include "AlpsWeightsReconstruction.h"
#include "AlpsSGD.h"
#include "AlpsAdaGrad.h"
#include "AlpsAdaDelta.h"
#include "AlpsAdam.h"
#include "AlpsActivations.h"
#include "AlpsCostFunction.h"
#include "AlpsConvolutionLayer.h"
#include "AlpsReconstructionLayer.h"
//
const int Dim = 2;
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
  //
  // The different strategy of gradient descent are:
  // - Alps::SGD - Online Gradient Descent
  // - Alps::AdaGrad - Adaptive Online Gradient Descent
  // - Alps::AdaDelta - Adaptive Delta Gradient Descent
  using Sigmoid          = Alps::Activation_sigmoid< double >;
  using Tanh             = Alps::Activation_tanh< double >;
  using ReLU             = Alps::Activation_ReLU< double >;
  using Kernel           = Alps::Window< double, Dim >;
  using Weights          = Alps::WeightsConvolution< double, Kernel, Alps::Arch::CPU, Tanh, Alps::AdaDelta, Dim >;
  using Weights_T        = Alps::WeightsTransposedConvolution< double, Kernel, Alps::Arch::CPU, Tanh, Alps::AdaDelta, Dim >;
  using ReconWeights     = Alps::WeightsReconstruction< double, Alps::Arch::CPU, Sigmoid, Alps::AdaDelta, Dim >;
  using LossFunction     = Alps::LeastSquarreEstimate< double >;
  using Convolutional    = Alps::ConvolutionLayer< Tanh, Weights, Kernel, LossFunction, Dim >;
  using Deconvolutional  = Alps::ConvolutionLayer< Tanh, Weights_T, Kernel, LossFunction, Dim >;
  using Reconstruction   = Alps::ReconstructionLayer< Sigmoid, ReconWeights, LossFunction, Dim >;

  //////////////////////////
  // Convolutional layers //
  //////////////////////////
//  //
//  // layer down level 1 - d10
//  // Window definition for the kernels
//  std::vector< long int > h_window_d10 = {0,0}; // size of the 1/2 window, when the value is 0, the kernel is the size of a pixel
//  std::vector< long int > padding_d10  = {0,0}; // padding
//  std::vector< long int > striding_d10 = {1,1}; // striding
//  //
//  Kernel window_d10( 16, // number of kernels
//		     h_window_d10, padding_d10, striding_d10 );
//  //
//  std::shared_ptr< Alps::Layer > nn_d10 =
//    std::make_shared< Convolutional >( "layer_d10",
//				       std::make_shared<Kernel>(window_d10) );
//  nn_d10->add_layer( nullptr );   // connection with the previous layer. (nullptr) means input layer
  //
  // layer down level 1 - d11
  // Window definition for the kernels
  std::vector< long int > h_window_d11 = {1,1}; // size of the 1/2 window
  std::vector< long int > padding_d11  = {1,1}; // padding
  std::vector< long int > striding_d11 = {1,1}; // striding
  //
  Kernel window_d11( 32, // number of kernels
		     h_window_d11, padding_d11, striding_d11 );
  //
  std::shared_ptr< Alps::Layer > nn_d11 =
    std::make_shared< Convolutional >( "layer_d11",
				       std::make_shared<Kernel>(window_d11) );
  nn_d11->add_layer( nullptr );   // connection with the previous layer. (nullptr) means input layer
//  //
//  // layer down level 1 - d12
//  // Window definition for the kernels. This window is centered on one voxel.
//  // If the padding and stridding is the same as the window 1, the output image
//  // will have exactly the same dimension.
//  std::vector< long int > h_window_d12 = {3,3}; // size of the 1/2 window
//  std::vector< long int > padding_d12  = {3,3}; // padding
//  std::vector< long int > striding_d12 = {1,1}; // striding
//  //
//  Kernel window_d12( 32, // number of kernels
//		     h_window_d12, padding_d12, striding_d12 );
//  //
//  std::shared_ptr< Alps::Layer > nn_d12 =
//    std::make_shared< Convolutional >( "layer_d12",
//				       std::make_shared<Kernel>(window_d12) );
//  nn_d12->add_layer( nullptr ); // connection with the previous layer. (nullptr) means input layer
  //
  // layer down level 2 - d20
  // Window definition for the kernels. This window is centered on one voxel.
  std::vector< long int > h_window_d20 = {2,2}; // size of the 1/2 window
  std::vector< long int > padding_d20  = {2,2}; // padding
  std::vector< long int > striding_d20 = {2,2}; // striding
  //
  Kernel window_d20( 64, // number of kernels
		     h_window_d20, padding_d20, striding_d20 );
  //
  std::shared_ptr< Alps::Layer > nn_d20 =
    std::make_shared< Convolutional >( "layer_d20",
				       std::make_shared<Kernel>(window_d20) );
//  nn_d20->add_layer( nn_d10 ); 
  nn_d20->add_layer( nn_d11 ); 
//  nn_d20->add_layer( nn_d12 );
//  //
//  // layer down level 2 - d21
//  // Window definition for the kernels. This window is centered on one voxel.
//  std::vector< long int > h_window_d21 = {3,3}; // size of the 1/2 window
//  std::vector< long int > padding_d21  = {3,3}; // padding
//  std::vector< long int > striding_d21 = {2,2}; // striding
//  //
//  Kernel window_d21( 32, // number of kernels
//		     h_window_d21, padding_d21, striding_d21 );
//  //
//  std::shared_ptr< Alps::Layer > nn_d21 =
//    std::make_shared< Convolutional >( "layer_d21",
//				       std::make_shared<Kernel>(window_d21) );
////  nn_d21->add_layer( nn_d10 ); 
//  nn_d21->add_layer( nn_d11 ); 
////  nn_d21->add_layer( nn_d12 );
  //
  // layer down level 3 - d30
  // Window definition for the kernels
  std::vector< long int > h_window_d30 = {3,3}; // size of the 1/2 window
  std::vector< long int > padding_d30  = {0,0}; // padding
  std::vector< long int > striding_d30 = {2,2}; // striding
  //
  Kernel window_d30( 128, // number of kernels
		     h_window_d30, padding_d30, striding_d30 );
  //
  std::shared_ptr< Alps::Layer > nn_d30 =
    std::make_shared< Convolutional >( "layer_d30",
				       std::make_shared<Kernel>(window_d30) );
  nn_d30->add_layer( nn_d20 );   // inputs and nn_2 are inputs of nn_1, then nn_1 is input of nn_4
//  nn_d30->add_layer( nn_d21 );   // inputs and nn_2 are inputs of nn_1, then nn_1 is input of nn_4



  ////////////////////////////
  // Deconvolutional layers //
  ////////////////////////////
  //
  // layer up level 3 - u30
  //
  Kernel window_u30( 128, // number of kernels
		     h_window_d30, padding_d30, striding_d30 );
  //
  std::shared_ptr< Alps::Layer > nn_u30 =
    std::make_shared< Deconvolutional >( "layer_u30",
					 std::make_shared<Kernel>(window_u30) );
  nn_u30->add_layer( nn_d30 );
//  //
//  // layer up level 2 - u21
//  //
//  Kernel window_u21( 32, // number of kernels
//		     h_window_d21, padding_d21, striding_d21 );
//  //
//  std::shared_ptr< Alps::Layer > nn_u21 =
//    std::make_shared< Deconvolutional >( "layer_u21",
//					 std::make_shared<Kernel>(window_u21) );
//  nn_u21->add_layer( nn_u30 );   
  //
  // layer up level 2 - u20
  //
  Kernel window_u20( 64, // number of kernels
		     h_window_d20, padding_d20, striding_d20 );
  //
  std::shared_ptr< Alps::Layer > nn_u20 =
    std::make_shared< Deconvolutional >( "layer_u20",
					 std::make_shared<Kernel>(window_u20) );
  nn_u20->add_layer( nn_u30 );   
//  //
//  // layer up level 1 - u12
//  //
//  Kernel window_u12( 32, // number of kernels
//		     h_window_d12, padding_d12, striding_d12 );
//  //
//  std::shared_ptr< Alps::Layer > nn_u12 =
//    std::make_shared< Deconvolutional >( "layer_u12",
//					 std::make_shared<Kernel>(window_u12) );
//  nn_u12->add_layer( nn_u20 );   
//  nn_u12->add_layer( nn_u21 );
  //
  // layer up level 1 - u11
  //
  Kernel window_u11( 32, // number of kernels
		     h_window_d11, padding_d11, striding_d11 );
  //
  std::shared_ptr< Alps::Layer > nn_u11 =
    std::make_shared< Deconvolutional >( "layer_u11",
					 std::make_shared<Kernel>(window_u11) );
  nn_u11->add_layer( nn_u20 );   
//  nn_u11->add_layer( nn_u21 );
//  //
//  // layer up level 1 - u10
//  //
//  Kernel window_u10( 16, // number of kernels
//		     h_window_d10, padding_d10, striding_d10 );
//  //
//  std::shared_ptr< Alps::Layer > nn_u10 =
//    std::make_shared< Deconvolutional >( "layer_u10",
//					 std::make_shared<Kernel>(window_u10) );
//  nn_u10->add_layer( nn_u20 );   
//  nn_u10->add_layer( nn_u21 );

  
  ////////////////////
  // Reconstruction //
  ////////////////////
  //
  // The output of nn_{7,8} should be combined togther as a linear combinaison in
  // a non-linear function, e.g. sigmoid, with one bias:
  // nn_9 = sigmoid( nn_7 + nn_8 + b). Then nn_9 should be directly compared to
  // the target in the cost function.
  std::shared_ptr< Alps::Layer > nn_reconstruction  =
    std::make_shared< Reconstruction >( "__output_layer__" );
//  nn_reconstruction->add_layer( nn_u10 );   // nn_7 are inputs of nn_8
  nn_reconstruction->add_layer( nn_u11 );   // nn_8 are inputs of nn_8
//  nn_reconstruction->add_layer( nn_u12 );   // nn_8 are inputs of nn_8


  
  
  /////////////
  // Anatomy //
  /////////////
//  mr_nn_.add( nn_d10 );
  mr_nn_.add( nn_d11 );
//  mr_nn_.add( nn_d12 );
  mr_nn_.add( nn_d20 );
//  mr_nn_.add( nn_d21 );
  mr_nn_.add( nn_d30 );
  mr_nn_.add( nn_u30 );
  mr_nn_.add( nn_u20 );
//  mr_nn_.add( nn_u21 );
//  mr_nn_.add( nn_u10 );
  mr_nn_.add( nn_u11 );
//  mr_nn_.add( nn_u12 );
  mr_nn_.add( nn_reconstruction );
};
//
//
//
void
Alps::Mont_Maudit_builder::forward( std::shared_ptr< Alps::Climber > Sub )
{
  mr_nn_.forward( Sub );
  // Down to subject
  std::shared_ptr< Alps::Subject< Dim > > subject = std::dynamic_pointer_cast< Alps::Subject< Dim > >(Sub);
  energy_subject_.push_back( subject->get_energy() );
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
Alps::Mont_Maudit_builder::save_weight_file( const std::size_t Epoque ) const
{
  //
  //
  std::string matrices_weights = "Mont_Maudit_" + std::to_string( Epoque ) + ".dat";
  std::ofstream weights_file( matrices_weights, std::ios::out | std::ios::binary | std::ios::trunc );
  // Cover the layers
  if ( weights_file.is_open() )
    mr_nn_.save_weights( weights_file );
  else
    {
      std::string mess = "The weights file has no I/O access.\n";
      throw MAC::MACException( __FILE__, __LINE__,
			       mess.c_str(),
			       ITK_LOCATION );
    }
  // close the file
  weights_file.close();
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
