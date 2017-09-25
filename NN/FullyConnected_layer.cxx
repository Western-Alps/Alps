//
//
//
#include "MACException.h"
#include "FullyConnected_layer.h"

/**
 * CUDA Kernel Device code
 *
 * Computes ...
 */
//__global__ void
//sqrt_cuda( double *A, int numElements)
//{
//    int i = blockDim.x * blockIdx.x + threadIdx.x;
//
//    if (i < numElements)
//    {
//      
//        A[i] = A[i] * A[i];
//	/*printf("lalla %f", A[i]);*/
//    }
//}


//
//
//
MAC::FullyConnected_layer::FullyConnected_layer( const std::string Layer_name,
						 const int         Layer_number,
						 const int         Number_fc_layers,
						 const int*        Fc_layers ):
  MAC::NeuralNetwork::NeuralNetwork(),
  layer_name_{Layer_name}, layer_number_{Layer_number},
  number_fc_layers_{Number_fc_layers}, fc_layers_{Fc_layers}
{
  //
  // number of weights
  for ( int w = 0 ; w < Number_fc_layers - 1 ; w++ )
    {
      number_of_weights_ += Fc_layers[w] * Fc_layers[w+1];
      number_of_neurons_ += Fc_layers[w];
    }
  // last layer
  number_of_neurons_ += Fc_layers[Number_fc_layers-1];

  //
  // Neurons
  activation_ = new double[number_of_neurons_];
  neurons_    = new double[number_of_neurons_];
};
//
//
//
void
MAC::FullyConnected_layer::initialization()
{
};
//
//
//
void
MAC::FullyConnected_layer::forward( Subject& Sub, const Weights& W )
{
};
//
//
//
MAC::FullyConnected_layer::~FullyConnected_layer()
{
  //
  delete[] activation_;
  activation_ = nullptr;
  //
  delete[] neurons_;
  neurons_    = nullptr;
};
