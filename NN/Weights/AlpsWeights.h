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
  enum class Arch
    {
     UNKNOWN   = -1,
     // 
     CPU       = 1,
     CUDA      = 2,  // NVIDIA GPU implementation
     CL        = 3   // heterogeneous computing
    }; 
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
    // Activation tensor from the previous layer
    virtual void set_activations( const std::vector< Alps::LayerTensors< Tensor1_Type, 2 > >&,
				  const std::vector< Alps::LayerTensors< Tensor1_Type, 2 > >& ) = 0;
    
    //
    // Functions
    //
    // Activate
    virtual std::tuple < std::shared_ptr< Tensor1_Type >,
			 std::shared_ptr< Tensor1_Type > > activate( std::vector< Alps::LayerTensors< Tensor1_Type, 2 > >& )       = 0;
    // Weighted error
    virtual void                                           weighted_error( std::vector< Alps::LayerTensors< Tensor1_Type, 2 > >&,
									   std::vector< Alps::LayerTensors< Tensor1_Type, 2 > >& ) = 0;
    // solver
    virtual void                                           solve()                                                                 = 0;
  };
}
#endif
