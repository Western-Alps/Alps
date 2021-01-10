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
  class WeightsFclCPU : public Alps::Weights< Eigen::MatrixXd, 2 >,
			public Alps::Climber
  {
  public:
    /** Constructor. */
    explicit WeightsFclCPU( std::shared_ptr< Alps::Mountain >,
			    const std::vector< std::size_t >,
			    const std::vector< std::size_t > );
    /** Destructor */
    virtual ~WeightsFclCPU(){};


    //
    // Accessors
    //
    // Get size of the tensor
    virtual const std::vector< std::size_t >      get_tensor_size()const                           override
    { return std::vector< std::size_t >(); };
    // Get the tensor
    virtual std::shared_ptr< Eigen::MatrixXd >    get_tensor() const                               override
    {return weights_;};
    // Set size of the tensor
    virtual void                                  set_tensor_size( std::vector< std::size_t > )    override{};
    // Set the tensor
    virtual void                                  set_tensor( std::shared_ptr< Eigen::MatrixXd > ) override{};
    //
    //
    // Get the observed mountain
    virtual std::shared_ptr< Alps::Mountain >     get_mountain()                                   override
    {return layer_;};

    
    //
    // Functions
    //
    // Save the weights
    virtual void                            save_tensor()const                                     override{};
    // Load the weights
    virtual void                            load_tensor( const std::string )                       override{};
    //
    //
    // Activate
    virtual std::shared_ptr< double >       activate( std::vector< Alps::LayerTensors< double, 2 > >&,
						      std::shared_ptr< Alps::Function > )          override;
    //
    //
    // Update the weights
    virtual void                            update()                                               override{};



  private:
    // Matrix of weigths
    std::shared_ptr< Eigen::MatrixXd > weights_;
    //
    // The mountain observed: fully connected layer
    std::shared_ptr< Alps::Mountain >  layer_;
  };
}
#endif
