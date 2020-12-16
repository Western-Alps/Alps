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
#include "AlpsClimber.h"
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
  class WeightsFclCPU : public Weights, public Alps::Climber
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
    // Overrrided accessors
    virtual std::shared_ptr< Eigen::MatrixXd > get_weight()            const override   
      {return weights_;};
    virtual std::shared_ptr< Eigen::MatrixXd > get_weight_transposed() const override
      {return weights_transposed_;};

    //
    // Function overrided
    virtual void save_weights()                                        const override {};
    // Save the weights
    virtual void load_weights()                                              override {};
    // Update the weights
    virtual void update()                                                    override {};

  private:
    // The mountain observed: fully connected layer
    std::shared_ptr< Alps::Mountain >  layer_;
    //
    std::shared_ptr< Eigen::MatrixXd > weights_;
    std::shared_ptr< Eigen::MatrixXd > weights_transposed_;
  };
}
#endif
