#ifndef MONTE_ROSA_BUILDER_H
#define MONTE_ROSA_BUILDER_H
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

  /** \class Monte_Rosa_builder
   *
   * \brief 
   * 
   * 
   */
  class Monte_Rosa_builder : public Alps::NeuralNetwork,
			     public Alps::Mountain
    {
    public:
      /** Constructor. */
      explicit Monte_Rosa_builder();
      /** Destructor */
      virtual ~Monte_Rosa_builder(){};

      
      //
      //
      // Accessors
      //
      // get the layer identification
      virtual const std::size_t             get_layer_id() const                                override
      { return layer_id_;}								        
      // get the layer name								        
      virtual const std::string             get_layer_name() const                              override
        { return std::string("Monte Rosa network.");};				        
      // Get number of weigths								        
      virtual const int                      get_number_weights() const                         override
        { return 1;};									        
      // get the layer size								        
      virtual const std::vector<std::size_t> get_layer_size() const                             override
      { return std::vector<std::size_t>();};						        
      // attach the next layer								        
      virtual void                           set_next_layer( std::shared_ptr< Alps::Layer > )   override{};
      //										        
      //										        
      // get the neural network energy							        
      virtual const double                   get_energy() const                                 override
      { return energy_.back();};
      // get the neural network energy							        
      virtual void                           set_energy( const double E )                       override
      { energy_.push_back( E );};


      //
      // Functions
      //
      // Add previous layer
      virtual void                           add_layer( std::shared_ptr< Alps::Layer > )        override{};
      // Forward propagation
      virtual void                           forward( std::shared_ptr< Alps::Climber > )        override;
      // Backward propagation
      virtual void                           backward( std::shared_ptr< Alps::Climber > )       override;
      // Update the weights at the end of the epoque
      virtual void                           weight_update( std::shared_ptr< Alps::Climber > )  override;
      //
      //
      // Add network layers
      virtual void                           add( std::shared_ptr< Alps::Layer > )              override{};
      //
      //
      // Overriding from Mountains
      virtual void                           attach( std::shared_ptr< Alps::Climber > C )       override
      { attached_climber_ = C;};
      // Notify the observers for updates
      virtual void                           notify()                                           override;
      // Save the weights at the end of the epoque
      virtual       void                     save_weight_file( const std::size_t ) const        override{};

      

    private:
      //
      //
      std::weak_ptr< Alps::Climber >       attached_climber_;

      //
      // layer unique ID
      std::size_t                          layer_id_{0};
      // energy of a subject per epoque
      std::vector< double>                 energy_subject_;
      // energy of the epoque
      std::vector< double >                energy_;
      //
      //
      NeuralNetworkComposite mr_nn_;
    };
}
#endif
