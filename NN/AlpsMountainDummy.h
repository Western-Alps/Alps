#ifndef ALPMOUNTAINDUMMY_H
#define ALPMOUNTAINDUMMY_H
//
//
//
#include <iostream> 
#include <memory>
//
// 
//
#include "AlpsClimber.h"
#include "AlpsMountain.h"
#include "AlpsNeuralNetwork.h"
#include "MACException.h"
//
//
//
/*! \namespace Alps
 *
 * Name space Alps.
 *
 */
namespace Alps
{
  /*! \class Mountain
   *
   * \brief class MountainDummy is a dummy instanciation of the interface for tests.
   * 
   * This pattern will be observed by the climbers (observers), e.g. waiting for
   * updates.
   *
   */
  // Observed (subject)
  class MountainDummy : public Alps::NeuralNetwork,
			public Alps::Mountain
  {
  public:
    
    //
    MountainDummy();
    //
    virtual ~MountainDummy(){};

    
    //
    // Accessors
    //
    // get the layer identification
    virtual const std::size_t      get_layer_id() const                              override
    { return -1;}
    // get the layer name
    virtual const std::string      get_layer_name() const                            override
    { return std::string("Mountain for Dummies.");};
    // Get number of weigths
    virtual const int              get_number_weights() const                        override
    { return 1;};
    // get the layer size
    virtual const std::vector<std::size_t> get_layer_size() const                    override
    { return std::vector<std::size_t>();};
    // attach the next layer
    virtual       void             set_next_layer( std::shared_ptr< Alps::Layer > )  override{};
    //
    //
    // get neural network energy
    virtual const double           get_energy() const                                override
    { return energy_;};
    // set neural network energy
    virtual void                   set_energy( const double E )                      override
    { energy_ = E;};

    
    //
    // functions
    //
    // Add previous layer
    virtual       void             add_layer( std::shared_ptr< Alps::Layer > )       override{};
    // Forward propagation
    virtual       void             forward( std::shared_ptr< Alps::Climber > )       override;
    // Backward propagation
    virtual       void             backward( std::shared_ptr< Alps::Climber > )      override;
    // Backward propagation
    virtual       void             weight_update( std::shared_ptr< Alps::Climber > ) override{};
    // Add network layers
    virtual       void             add( std::shared_ptr< Alps::Layer > )             override{};
    //
    // Attach observers that need to be updated
    virtual       void             attach( std::shared_ptr< Alps::Climber > C )      override{};
    // Notify the observers for updates
    virtual       void             notify()                                          override{};


  private:
    std::weak_ptr< Alps::Climber > attached_climber_;
    double energy_{10.};
  };
}
#endif

