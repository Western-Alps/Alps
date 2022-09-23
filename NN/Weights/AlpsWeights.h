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
  //! Architecture
  /*! Represents the type of device architecture processing the weights. */
  enum class Arch
    {
     UNKNOWN   = -1,
     // 
     CPU       = 1, /*!< Central Processing Unit. */
     CUDA      = 2, /*!< CUDA on NVIDIA Graphical Processing Unit.*/
     CL        = 3  /*!< OpenCL on Graphical Processing Unit.*/
    }; 
  /*! \class Weights
   * \brief class representing the weights container used by all the 
   * neural networks layers. 
   *
   */
  template< typename Tensor1_Type, typename Tensor2_Type, int Dim >
  class Weights : public Alps::Tensor< Tensor2_Type, /*Order*/ 2 >
  {
    //
    // Aliases
    using LayerTensorsVec = std::vector< std::reference_wrapper< Alps::LayerTensors< Tensor1_Type, Dim > > >;
    using ActivationVec   = std::array < std::vector< Tensor1_Type >, 2 >;

    
  public:
    // Destructor
    virtual ~Weights() = default;


    //
    // Accessors
    //
    //! Activation tensor from the previous layer
    /*!
      Process ...
    */
    virtual void           set_activations( LayerTensorsVec&,
					    LayerTensorsVec& )  = 0;
    
    //
    // Functions
    //
    //! Activate
    /*!
      Process the activation and the derivative from the layer.
      \param Load the previouse layer tensors
      \return an array with the two first tensors
    */
    virtual ActivationVec  activate( LayerTensorsVec& )         = 0;
    //! Weighted error
    /*!
      Process the weithed error for other layers.
      \param Load the previouse layer tensors
      \param Load the previouse layer tensors
    */
    virtual void           weighted_error( LayerTensorsVec&,
					   LayerTensorsVec& )   = 0;
    //! Update the weights
    /*!
      Update the weights during the backward propagation process.
    */
    virtual void           update()                             = 0;
    //! Force the update of the weights
    /*!
      Force the update the weights during the backward propagation process.
    */
    virtual void           forced_update()                      = 0;
  };
}
#endif
