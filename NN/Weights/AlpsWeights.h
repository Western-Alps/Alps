#ifndef ALPSWEIGHTS_H
#define ALPSWEIGHTS_H
//
//
//
#include <iostream> 
#include <memory>
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
//
//
//
#include "MACException.h"
#include "AlpsTensor.h"
#include "AlpsLayerTensors.h"
#include "AlpsImage.h"
#include "AlpsFunction.h"
/*! \namespace Alps
 *
 * Name space for Alps.
 *
 */
namespace Alps
{
  /*! \class Weights
   * \brief class representing the weights container used by all the 
   * neural networks layers. 
   *
   */
  template< typename Type, int Order >
  class Weights : public Alps::Tensor< Type, Order >
  {
  public:
    // Destructor
    virtual ~Weights(){};


    //
    // Accessors
    //

    
    //
    // Functions
    //
    // Activate
    virtual std::shared_ptr< double > activate( std::vector< Alps::LayerTensors< double, 2 > >&,
						std::shared_ptr< Alps::Function > ) = 0;
  };
}
#endif
