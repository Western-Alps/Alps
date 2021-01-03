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
  template< typename Type >
  class Weights : public Alps::Tensor< Type >
  {
  public:
    // Destructor
    virtual ~Weights(){};


    //
    // Accessors
    //
    // Get the weigths matrix
    virtual const std::vector< Eigen::MatrixXd >& get_weights()                                              const = 0;

    //
    // Functions
    //
    // Activate
    virtual void                                  activate( std::vector< std::shared_ptr< Type > >& )              = 0;
  };
}
#endif
