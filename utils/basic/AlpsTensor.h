/*=========================================================================
* Alps is a deep learning library approach customized for neuroimaging data 
* Copyright (C) 2021 Yann Cobigo (yann.cobigo@yahoo.com)
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <https://www.gnu.org/licenses/>.
*=========================================================================*/
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
   * of our problems should fit these two choices.
   * 
   */
  template< typename Type, int Order >
  class Tensor
    {
      //
      // 
    public:
      /** Destructor */
      virtual ~Tensor() = default;


      
      //
      // Accessors
      //
      //! Get size of the tensor
      /*!
	\return a vector with the size of each dimensions
      */
       virtual const std::vector< std::size_t > get_tensor_size() const noexcept     = 0;
      //! Get the tensor
      /*!
	\return a vector tensor
      */
      virtual const std::vector< Type >&        get_tensor() const noexcept          = 0;
      //! Update the tensor
      /*!
	\return the updated vector tensor
      */
      virtual std::vector< Type >&              update_tensor()                      = 0;


      
      //
      // Functions
      //
      //! Save the tensor values (e.g. weights)
      virtual void                              save_tensor( std::ofstream& ) const  = 0;
      //! Load the tensor values (e.g. weights)
      /*!
	\param name of the saved tensor
      */
      virtual void                              load_tensor( const std::ofstream& )  = 0;
    };
}
#endif
