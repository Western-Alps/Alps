//
// CUDA
//
//
//
#include "MACException.h"
#include "NeuralNetwork.cuh"
#include "NeuralNetworkComposite.cuh"
#include "Monte_Rosa_builder.cuh"
//
//
//
MAC::Monte_Rosa_builder::Monte_Rosa_builder():
  MAC::NeuralNetwork::NeuralNetwork()
{
};
//
//
//
__host__ void
MAC::Monte_Rosa_builder::forward()
{
  mr_nn_.forward();
};
