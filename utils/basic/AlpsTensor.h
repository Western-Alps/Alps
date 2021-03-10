#ifndef ALPSTENSOR_H
#define ALPSTENSOR_H
//
//
// ITK
#include "ITKHeaders.h"
//
#include <iostream>
#include <memory>
#include "MACException.h"
//
//
//
namespace Alps
{
  /** \class Tensor
   *
   * \brief 
   * Tensor object represents the basic memory element for the neural network.
   * Type:
   *   - float  (e.g.) the tensor is one dimension  (1D).
   *   - float* (e.g.) the tensor is two dimensions (2D).
   *
   * We do not implement tensors with more than 2D and less then 1D (scalars). Most
   * of our problems should fit these two choices. /
   * 
   */
  template< typename Type, int Order >
  class Tensor
    {
      //
      // 
    public:
      /** Destructor */
      virtual ~Tensor(){};

      //
      // Accessors
      //
      // Get size of the tensor
      virtual const std::vector< std::size_t > get_tensor_size() const                       = 0;
      // Get the tensor
      virtual std::shared_ptr< Type >          get_tensor() const                            = 0;
      // Set size of the tensor
      virtual void                             set_tensor_size( std::vector< std::size_t > ) = 0;
      // Set the tensor
      virtual void                             set_tensor( std::shared_ptr< Type > )         = 0;

      //
      // Functions
      //
      // Save the tensor values (e.g. weights)
      virtual void                             save_tensor() const                           = 0;
      // Load the tensor values (e.g. weights)
      virtual void                             load_tensor( const std::string )              = 0;
    };
}
#endif
