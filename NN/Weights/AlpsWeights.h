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
#include "AlpsLayerDependencies.h"
#include "MACException.h"
#include "AlpsClimber.h"
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
  class Weights
  {
//  protected:
//    Weights(){};
  public:
    // Destructor
    virtual ~Weights(){};


  public:
    //
    // Overrrided accessors
    virtual std::shared_ptr< Eigen::MatrixXd > get_weight()            const = 0;
    virtual std::shared_ptr< Eigen::MatrixXd > get_weight_transposed() const = 0;
    // Save the weightd
    virtual void save_weights()                                        const = 0;
    // Save the weight
    virtual void load_weights()                                              = 0;
  };
}
#endif
