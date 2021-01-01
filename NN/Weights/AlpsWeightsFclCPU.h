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
#include "AlpsClimber.h"
#include "AlpsMountain.h"
//
//
//
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
  class WeightsFclCPU : public Alps::Weights,
			public Alps::Climber
  {
  public:
    /** Constructor. */
    explicit WeightsFclCPU( std::shared_ptr< Alps::Mountain >,
			    const std::vector< int >,
			    const std::vector< int > );
    /** Destructor */
    virtual ~WeightsFclCPU(){};


    //
    // Accessors
    //
    // Get the weigths matrix
    virtual const std::vector< Eigen::MatrixXd >& get_weights()            const override   
      {return weights_;};
    // Get the transposed weights matrix
    virtual const std::vector< Eigen::MatrixXd >& get_weights_transposed() const override
      {return weights_transposed_;};


    //
    // Functions
    //
    // Save the weights
    virtual void                                  save_weights()           const override {};
    // Load the weights
    virtual void                                  load_weights()                 override {};
    //
    //
    // Get the observed mountain
    virtual std::shared_ptr< Alps::Mountain >     get_mountain()                 override
    {return layer_;};
    // Update the weights
    virtual void                                  update()                       override {};



  private:
    // Matrix of weigths
    std::vector< Eigen::MatrixXd >    weights_;
    // Transposed matrix of weigths
    std::vector< Eigen::MatrixXd >    weights_transposed_;
    //
    // The mountain observed: fully connected layer
    std::shared_ptr< Alps::Mountain > layer_;
  };
}
#endif
