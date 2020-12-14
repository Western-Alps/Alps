#ifndef ALPSWEIGHTSFCLCPU_H
#define ALPSWEIGHTSFCLCPU_H
//
//
//
#include <iostream> 
#include <vector>
#include <memory>
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
//
#include "MACException.h"
#include "AlpsWeights.h"
#include "AlpsMountain.h"
/*! \namespace Alps
 *
 * Name space for Alps.
 *
 */
namespace Alps
{
  // Forward declaration of Conatainer
  
  /*! \class WeightsFullyConnected
   * \brief class representing the weights container for fully
   * connected layers (FCL) using CPU.
   *
   */
  class WeightsFclCPU : public Weights
  {
  public:
    // Costructor
    WeightsFclCPU( std::shared_ptr< Alps::Mountain >,
		   const std::vector< int >,
		   const std::vector< int > );
    // Destructor
    virtual ~WeightsFclCPU(){};


  public:
    //
    // Save the weights
    virtual void save_weights() const override {};
    // Save the weights
    virtual void load_weights()       override {};
    // Save the weights
    virtual void update()             override {};

  private:
    // The mountain observed: fully connected layer
    std::shared_ptr< Alps::Mountain > layer_;
    //
    Eigen::MatrixXd                   weights_;
    Eigen::MatrixXd                   weights_transposed_;
  };
}
#endif
