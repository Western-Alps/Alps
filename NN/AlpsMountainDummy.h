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
   * This pattern will be observed by the climbers (observers) for instant 
   * updates.
   *
   */
  // Observed (subject)
  class MountainDummy : public Alps::NeuralNetwork, public Alps::Mountain
  {
  public:
    
    //
    MountainDummy();
    //
    virtual ~MountainDummy(){};
    
    //
    // Accessors
    
    
    //
    // functions
    //
    // get the layer name
    virtual const std::string get_layer_name()                            const override
    { return std::string("Mountain for Dummies.");};
    // get the layer name
    virtual const double      get_energy()                                const override
    { return energy_;};
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
    // Attach observers that need to be updated
    virtual       void        attach( std::shared_ptr< Alps::Climber > )        override {};
    // Notify the observers for updates
    virtual       void        notify()                                          override {};

  private:
    //
    double energy_{10.};
  };
}
#endif

