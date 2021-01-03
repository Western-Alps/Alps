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
#include "AlpsImage.h"
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
    virtual void activate( std::vector< std::shared_ptr< Alps::Image< double, 1 > > >& ) = 0;
  };
}
#endif
