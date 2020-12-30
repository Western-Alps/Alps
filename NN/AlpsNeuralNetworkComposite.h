#ifndef ALPSNEURALNETWORKCOMPOSITE_H
#define ALPSNEURALNETWORKCOMPOSITE_H
//
//
//
#include <iostream>
#include <fstream>
#include <math.h>  
#include <map>
#include <list>
#include <memory>
//
// CUDA
//
#include <cuda_runtime.h>
//
//
//
#include "AlpsNeuralNetwork.h"
//
//
//
namespace Alps
{

  /** \class NeuralNetworkComposite
   *
   * \brief 
   * 
   * 
   */
  class NeuralNetworkComposite : public Alps::NeuralNetwork
    {
      
    public:
      /** Constructor. */
      NeuralNetworkComposite();
      /** Destructor */
      virtual ~NeuralNetworkComposite();

      
      //
      // Accessors
      //
      // get the layer name
      virtual const std::string      get_layer_name()                                 const override
        { return "composit layer";};
      // get number of weights
      virtual const int              get_number_weights()                             const override
      { return 0.;};
      // get the layer size
      virtual const std::vector<int> get_layer_size()                                 const override
      { return std::vector<int>();};
      // attach the next layer
      virtual       void             set_next_layer( std::shared_ptr< Alps::Layer > )       override{};
      //
      //
      // get neural network energy
      virtual const double           get_energy()                                     const override
      { return 0.;};


      //
      //
      // Forward propagation
      virtual       void             forward( std::shared_ptr< Alps::Climber > )            override;
      //
      virtual       void             backward()                                             override;
      //
      virtual       void             add( std::shared_ptr< Alps::Layer > )                  override;

    private:
      // Structure of the composite neural network
      std::list< std::shared_ptr< Alps::Layer > > nn_composite_;
    };
}
#endif
