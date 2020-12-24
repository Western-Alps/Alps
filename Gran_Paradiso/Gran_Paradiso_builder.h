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
  class Gran_Paradiso_builder : public Alps::NeuralNetwork, public Alps::Mountain
    {
    public:
      /** Constructor. */
      Gran_Paradiso_builder();
      //
      //explicit Subject( const int, const int );

      /** Destructor */
      virtual ~Gran_Paradiso_builder(){};

      //
      // Overriding from Network
      // Initialization
      //virtual void initialization();
      // get the layer name
      virtual const std::string get_layer_name()                            const override
        { return std::string("Gran Paradiso network.");};
      // get the layer name
      //virtual Layer get_layer_type(){ return Gran_Paradiso_layer;};
      // get the layer name
      virtual const double      get_energy()                                const override
        { return 1. /*ToDo*/;};
      // Get number of weigths
      virtual const int         get_number_weights()                        const override
        { return 1;};
      // Forward propagation
      virtual       void        forward( std::shared_ptr< Alps::Climber > )       override;
      // Backward propagation
      virtual       void        backward()                                        override;
      // Backward error propagation
      //virtual void backward_error_propagation(){};
      // Add network layers
      virtual       void        add( std::shared_ptr< Alps::Layer > )             override{};
      //
      // Overriding from Maountains
      virtual       void        attach( std::shared_ptr< Alps::Climber > )        override {};
      // Notify the observers for updates
      virtual       void        notify()                                          override {};

      

    private:
      //
      //
      Alps::NeuralNetworkComposite mr_nn_;
    };
}
#endif
