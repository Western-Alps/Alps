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
#include "AlpsBaseFunction.h"
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
  template< typename Tensor1_Type, typename Tensor2_Type >
  class Weights : public Alps::Tensor< Tensor2_Type, /*Order*/ 2 >
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
    virtual std::shared_ptr< Tensor1_Type > activate( std::vector< Alps::LayerTensors< Tensor1_Type, 2 > >& ) = 0;
  };
}
#endif
