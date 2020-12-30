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
    // get the layer name
    virtual const std::string      get_layer_name()                                 const override
    { return std::string("Mountain for Dummies.");};
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
    // get neural network energy
    virtual const double           get_energy()                                     const override
    { return energy_;};

    
    //
    // functions
    //
    // Forward propagation
    virtual       void             forward( std::shared_ptr< Alps::Climber > )            override;
    // Backward propagation
    virtual       void             backward()                                             override;
    // Add network layers
    virtual       void             add( std::shared_ptr< Alps::Layer > )                  override{};
    //
    // Attach observers that need to be updated
    virtual       void             attach( std::shared_ptr< Alps::Climber > )             override{};
    // Notify the observers for updates
    virtual       void             notify()                                               override{};


  private:
    double energy_{10.};
  };
}
#endif

