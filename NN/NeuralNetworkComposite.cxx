//
//
//
#include "MACException.h"
#include "NeuralNetworkComposite.h"

/**
 * CUDA Kernel Device code
 *
 * Computes ...
 */
//__global__ void
//cuda_NeuralNetworkComposite( double *A, int numElements)
//{
//  int i = blockDim.x * blockIdx.x + threadIdx.x;
//
//  if (i < numElements)
//    {
//      
//      A[i] = A[i] * A[i];
//      /*printf("lalla %f", A[i]);*/
//    }
//}
//
//
//
MAC::NeuralNetworkComposite::NeuralNetworkComposite():
  MAC::NeuralNetwork::NeuralNetwork(),
  weights_()
{
  try
    {
      //
      // Error code to check return values for CUDA calls

      //
//      //
//      int num_elem = 2000;
//      size_t size_of_test = num_elem * sizeof( double );
//
//      //
//      // Allocation
//      // Allocate the host
//      weights_   = (double *) malloc( size_of_test );
//      // Allocate the device 
//      d_weights_ = nullptr;
//      cuda_err_ = cudaMalloc( (void **) &d_weights_, size_of_test );
//      //
//      if ( cuda_err_ != cudaSuccess )
//	{
//	  std::string mess  = "Failed to allocate device; error code:\n";
//	  mess += cudaGetErrorString( cuda_err_ );
//	  //
//	  throw MAC::MACException( __FILE__, __LINE__,
//				   mess.c_str(),
//				   ITK_LOCATION );
//	}

      //
      // Neural network anatomy
      //
	    
//      //
//      // Create the data
//      for ( int i = 0 ; i < num_elem ; i++ )
//	weights_[i] = static_cast< double >( i );
//
//	    
//      //
//      // Copy the data
//      cuda_err_ = cudaMemcpy(d_weights_, weights_, size_of_test, cudaMemcpyHostToDevice);
//      if ( cuda_err_ != cudaSuccess )
//	{
//	  std::string mess  = "Failed to copy to device; error code:\n";
//	  mess += cudaGetErrorString( cuda_err_ );
//	  //
//	  throw MAC::MACException( __FILE__, __LINE__,
//				   mess.c_str(),
//				   ITK_LOCATION );
//	}
//
//
//      //
//      // 
//      std::shared_ptr< NeuralNetwork > nn_1 = std::make_shared< MAC::NN_test >( weights_,
//										d_weights_ );
//      std::shared_ptr< NeuralNetwork > nn_2 = std::make_shared< MAC::Convolutional_layer >( weights_,
//											    d_weights_ );
//      nn_composite_.push_back( nn_1 );
//      nn_composite_.push_back( nn_2 );

	    

	    
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << err << std::endl;
      exit(EXIT_FAILURE);
    }
};
//
//
//
void
MAC::NeuralNetworkComposite::initialization()
{
  try
    {
      //
      // Go through elements
      for ( auto nn_elem : nn_composite_ )
	{
	  std::cout << "Initialization elem: " << nn_elem->get_layer_name()
		    << std::endl;
	  // record the number of weights
	  switch( nn_elem->get_layer_type() )
	    {
	    case convolutional_layer:
	      {
		weight_indexes_.push_back( number_of_weights_ );
		number_of_weights_ += nn_elem->get_number_weights();
		break;
	      }
	    case fully_connected_layer:
	      {
		weight_indexes_.push_back( number_of_weights_ );
		number_of_weights_ += nn_elem->get_number_weights();
		break;
	      }
	    default:
	      throw MAC::MACException( __FILE__, __LINE__,
				       "Layer type unknown.",
				       ITK_LOCATION );
	    }
	}

      //
      // Init the weights
      weights_ = Weights( number_of_weights_, weight_indexes_ );
      // weights_.print();
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << err << std::endl;
      exit(EXIT_FAILURE);
    }
}
//
//
//
MAC::NeuralNetworkComposite::~NeuralNetworkComposite()
{
  try
    {
//      // free malloc
//      free( weights_ );
//      // free Cuda malloc
//      cuda_err_ = cudaFree( d_weights_ );
//      if ( cuda_err_ != cudaSuccess )
//	{
//	  std::string mess  = "Failed to allocate device; error code:\n";
//	  mess += cudaGetErrorString( cuda_err_ );
//	  //
//	  throw MAC::MACException( __FILE__, __LINE__,
//				   mess.c_str(),
//				   ITK_LOCATION );
//	}
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << err << std::endl;
      exit(EXIT_FAILURE);
    }
}
