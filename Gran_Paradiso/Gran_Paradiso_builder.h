#ifndef GRAN_PARADISO_BUILDER_H
#define GRAN_PARADISO_BUILDER_H
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
#include "MACException.h"
#include "AlpsSubject.h"
#include "AlpsMountain.h"
#include "AlpsNeuralNetwork.h"
#include "AlpsNeuralNetworkComposite.h"
//
//
//
namespace Alps
{

  /** \class Gran_Paradiso_builder
   *
   * \brief 
   * 
   * 
   */
  class Gran_Paradiso_builder : public Alps::NeuralNetwork,
				public Alps::Mountain
    {
    public:
      /** Constructor. */
      explicit Gran_Paradiso_builder();
      /** Destructor */
      virtual ~Gran_Paradiso_builder(){};

      
      //
      //
      // Accessors
      //
      // get the layer name
      virtual const std::string      get_layer_name()                                 const override
        { return std::string("Gran Paradiso network.");};
      // Get number of weigths
      virtual const int              get_number_weights()                             const override
        { return 1;};
      // get the layer size
      virtual const std::vector<int> get_layer_size()                                 const override
      { return std::vector<int>();};
      // attach the next layer
      virtual       void             set_next_layer( std::shared_ptr< Alps::Layer > )       override{};
      //
      //
      // get the neural network energy
      virtual const double           get_energy()                                     const override
        { return 1. /*ToDo*/;};


      //
      // Functions
      //
      // Forward propagation
      virtual       void        forward( std::shared_ptr< Alps::Climber > )                 override;
      // Backward propagation
      virtual       void        backward()                                                  override;
      //
      //
      // Add network layers
      virtual       void        add( std::shared_ptr< Alps::Layer > )                       override{};
      //
      //
      // Overriding from Mountains
      virtual       void        attach( std::shared_ptr< Alps::Climber > )                  override{};
      // Notify the observers for updates
      virtual       void        notify()                                                    override{};

      

    private:
      //
      //
      Alps::NeuralNetworkComposite mr_nn_;
    };
}
#endif
