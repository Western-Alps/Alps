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
  public:
    // Destructor
    virtual ~Weights(){};


    //
    // Accessors
    //
    // Get the weigths matrix
    virtual std::shared_ptr< Eigen::MatrixXd > get_weight()            const = 0;
    // Get the transposed weights matrix
    virtual std::shared_ptr< Eigen::MatrixXd > get_weight_transposed() const = 0;
    // Save the weights
    virtual void save_weights()                                        const = 0;
    // Load the weights
    virtual void load_weights()                                              = 0;
  };
}
#endif
