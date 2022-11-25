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
#ifndef ALPSLAYERTENSORS_H
#define ALPSLAYERTENSORS_H
//
//
//
#include <iostream> 
#include <algorithm>
//
// ITK
//
#include "ITKHeaders.h"
//
//
//
#include "MACException.h"
#include "AlpsImage.h"
#include "AlpsTools.h"
//
//
//
/*! \namespace Alps
 *
 * Name space Alps.
 *
 */
namespace Alps
{
  //! TensorOrder1
  /*! The N dimensions images are flatten into a 1D tensor. The main tensors used in the application are: */
  enum class TensorOrder1
    { 
     UNKNOWN    = -1,
     ACTIVATION =  0, /*!< The layer activation. */
     DERIVATIVE =  1, /*!< The layer activation derivation. */
     ERROR      =  2, /*!< The layer error calculated from other layers. */
     WERROR     =  3  /*!< The layer weighted error calculated from other layers. */
    };

  /*! \class LayerTensors
   *
   * \brief class LayerTensors records the Layer flatten images
   * (tensor order 1). It holds for an image i:
   *  - [0] Activation
   *  - [1] Derivative of the activation
   *  - [2] error from the layer
   *  - [3] weighted error from the layer
   *
   * And memorized the tensors from  previous epoques.
   *
   */
  template< typename Type, int Dim >
  class LayerTensors : public Alps::Tensor< Type, 1 >
  {
  public:
    /** Constructor */
    LayerTensors( const std::string );
    /** Constructor */
    LayerTensors( const std::vector< std::size_t >,
		  std::array< std::vector< Type >, 4 > );
    /** Constructor */
    LayerTensors( const std::array< std::size_t, Dim >,
		  std::array< std::vector< Type >, 4 > );
    /* Destructor */
    virtual ~LayerTensors() = default;

    
    //
    // Accessors
    //
    // From Alps::Tensor< Type, 1 >
    //
    //! Get size of the tensor
    virtual const std::vector< std::size_t >    get_tensor_size() const noexcept              override
    { return tensors_[0].get_tensor_size();};
    //! Get the tensor
    virtual const std::vector< Type >&          get_tensor() const noexcept                   override
    { return tensors_[0].get_tensor();};
    //! Update the tensor
    virtual std::vector< Type >&                update_tensor()                               override 
    { return tensors_[0].update_tensor();};
    //
    // From LayerTensors< typename Type, int Dim >
    //
    // Access the images directly
    Alps::Image< Type, Dim >&                   get_image( Alps::TensorOrder1 );

    
    //
    // Functions
    //
    // From Alps::Tensor< Type, 1 >
    //
    //! Save the tensor values (e.g. weights)
    virtual void                                save_tensor( std::ofstream& ) const           override{};
    //! Load the tensor values (e.g. weights)
    virtual void                                load_tensor( const std::ofstream& )           override{};
    //
    // From LayerTensors< typename Type, int Dim >
    //
    //! Implementation of [] operator.
    /*!
      This function retunrs one of the 4 application main tensor's element.
      \return an element
    */
    std::vector< Type >&                        operator[]( Alps::TensorOrder1 Idx ); 
    //! Implementation of () operator.
    /*!
      This function computs the Hamadard production of two 1D tensors.
      \param First 1D tensor for the Hamadard product
      \param Second 1D tensor for the Hamadard product
      \return a 1D tensor
    */
    std::vector< Type >                         operator()( Alps::TensorOrder1, Alps::TensorOrder1 ); 
    //
    void                                        replace( const std::vector< std::size_t >,
							  std::array< std::vector< Type >, 4 > );  
    //
    void                                        replace( const std::array< std::size_t, Dim >,
							  std::array< std::vector< Type >, 4 > );  
    
  private:
    // (Z,error, )
    std::array< Alps::Image< Type, Dim >, 4 >                tensors_;
    // 
    std::vector< std::array< Alps::Image< Type, Dim >, 4 > > previous_epoque_tensors_;
  };
  //
  //
  // Constructor
  template< typename T,int D >
  Alps::LayerTensors< T, D >::LayerTensors( const std::string Image )
  {
    try
      {
	//
	// load the image ITK pointer
	auto image_ptr = itk::ImageIOFactory::CreateImageIO( Image.c_str(),
							     itk::CommonEnums::IOFileMode::ReadMode );
	image_ptr->SetFileName( Image );
	image_ptr->ReadImageInformation();
	// Check the dimensions complies
	if ( image_ptr->GetNumberOfDimensions() != D )
	  throw MAC::MACException( __FILE__, __LINE__,
				   "The dimensions of the image and instanciation are different.",
				   ITK_LOCATION );
	//
	// Read the ITK image
	typename Reader< D >::Pointer img_ptr = Reader< D >::New();
	img_ptr->SetFileName( image_ptr->GetFileName() );
	img_ptr->Update();
	//
	// Load the modalities into the container
	tensors_[0] = Alps::Image< double, D >( img_ptr );
	// Load the other tensors
	std::vector< std::size_t > tensor_size = tensors_[0].get_tensor_size();
	tensors_[1] = Alps::Image< T, D >( tensor_size, std::vector<T>(tensor_size[0], 0.) );
	tensors_[2] = Alps::Image< T, D >( tensor_size, std::vector<T>(tensor_size[0], 0.) );
	tensors_[3] = Alps::Image< T, D >( tensor_size, std::vector<T>(tensor_size[0], 0.) );
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
      }
  }
  //
  //
  // Constructor
  template< typename T,int D >
  Alps::LayerTensors< T, D >::LayerTensors( const std::vector< std::size_t >  Tensor_size,
					    std::array< std::vector< T >, 4 > Tensors )
  {
    try
      {
	//
	// Load the modalities into the container
	for ( int t = 0 ; t < 4 ; t++ )
	  if ( Tensor_size[0] != Tensors[t].size() )
	    {
	      std::cout
		<< "Tensor_size[0] " << Tensor_size[0]
		<< " Tensors["<<t<<"].size() " << Tensors[t].size()
		<< std::endl;
		throw MAC::MACException( __FILE__, __LINE__,
				       "We can't build a layer with inappropriate size",
				       ITK_LOCATION );
	    }
	//
	tensors_[0] = Alps::Image< T, D >( Tensor_size, Tensors[0] );
	tensors_[1] = Alps::Image< T, D >( Tensor_size, Tensors[1] );
	tensors_[2] = Alps::Image< T, D >( Tensor_size, Tensors[2] );
	tensors_[3] = Alps::Image< T, D >( Tensor_size, Tensors[3] );
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
      }
  }
  //
  //
  // Constructor
  template< typename T,int D >
  Alps::LayerTensors< T, D >::LayerTensors( const std::array< std::size_t, D >   Tensor_size,
					    std::array< std::vector< T >, 4 >    Tensors )
  {
    try
      {
	//
	// Load the modalities into the container
	std::size_t t_size = 1.;
	for ( int d = 0 ; d < D ; d++ )
	  t_size *= Tensor_size[d];
	for ( int t = 0 ; t < 4 ; t++ )
	  if ( t_size != Tensors[t].size() )
	    {
	      std::cout
		<< "Tensor size = " << t_size
		<< " Tensors["<<t<<"].size() " << Tensors[t].size()
		<< std::endl;
		throw MAC::MACException( __FILE__, __LINE__,
					 "We can't build a layer with inappropriate size",
					 ITK_LOCATION );
	    }
	//
	tensors_[0] = Alps::Image< T, D >( Tensor_size, Tensors[0] );
	tensors_[1] = Alps::Image< T, D >( Tensor_size, Tensors[1] );
	tensors_[2] = Alps::Image< T, D >( Tensor_size, Tensors[2] );
	tensors_[3] = Alps::Image< T, D >( Tensor_size, Tensors[3] );
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
      }
  }
  //
  //
  //
  template< typename T,int D > Alps::Image< T, D >&
  Alps::LayerTensors< T, D >::get_image( Alps::TensorOrder1 Idx )
  {
    try
      {
	if ( static_cast< int >( Idx ) > 3 ||
	     static_cast< int >( Idx ) < 0 )
	  throw MAC::MACException( __FILE__, __LINE__,
				   "Indexing not implemented yet.",
				   ITK_LOCATION );
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
      }
    //
    //
    return tensors_[ static_cast< int >( Idx ) ]; 
  }
  //
  //
  // Operator []
  template< typename T,int D > std::vector< T >&
  Alps::LayerTensors< T, D >::operator[]( Alps::TensorOrder1 Idx ) 
  {
    try
      {
	if ( static_cast< int >( Idx ) > 3 ||
	     static_cast< int >( Idx ) < 0 )
	  throw MAC::MACException( __FILE__, __LINE__,
				   "Indexing not implemented yet.",
				   ITK_LOCATION );
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
      }
    //
    //
    return tensors_[ static_cast< int >( Idx ) ].update_tensor(); 
  }
  //
  //
  // Operator ()
  template< typename T,int D > std::vector< T >
  Alps::LayerTensors< T, D >::operator()( Alps::TensorOrder1 Idx1,
					  Alps::TensorOrder1 Idx2 ) 
  {
    try
      {
	if ( static_cast< int >( Idx1 ) > 3 ||
	     static_cast< int >( Idx1 ) < 0 ||
	     static_cast< int >( Idx2 ) > 3 ||
	     static_cast< int >( Idx2 ) < 0 )
	  throw MAC::MACException( __FILE__, __LINE__,
				   "Indexing not implemented yet.",
				   ITK_LOCATION );
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
      }
    //
    //
    std::size_t img_size = tensors_[ static_cast< int >( Idx1 ) ].get_tensor_size()[0];
    // prepare the return pointer
    std::vector< T > hadamard( img_size, 0. );
    //
    for ( std::size_t s = 0 ; s < img_size ; s++ )
      {
	hadamard[s] =
	  tensors_[ static_cast< int >( Idx1 ) ].get_tensor()[s] *
	  tensors_[ static_cast< int >( Idx2 ) ].get_tensor()[s];
      }
      
    //
    //
    return hadamard;
  }
  //
  //
  // 
  template< typename T,int D > void
  Alps::LayerTensors< T, D >::replace( const std::vector< std::size_t >  Tensor_size,
				       std::array< std::vector< T >, 4 > Tensors )
  {
    try
      {
	//
	//
	for ( int t = 0 ; t < 4 ; t++ )
	  if ( Tensor_size[0] != Tensors[t].size() )
	    throw MAC::MACException( __FILE__, __LINE__,
				     "Error on the tensor size.",
				     ITK_LOCATION );
	//
	// Save the previous set of neurons
	// ToDo: implement de move sementic in AlpsImage
	// ToDo: it cost a lot of memory: previous_epoque_tensors_.push_back( /*std::move(*/tensors_/*)*/ );
	// Load new tensors
	tensors_[0] = Alps::Image< T, D >( Tensor_size, Tensors[0] );
	tensors_[1] = Alps::Image< T, D >( Tensor_size, Tensors[1] );
	tensors_[2] = Alps::Image< T, D >( Tensor_size, Tensors[2] );
	tensors_[3] = Alps::Image< T, D >( Tensor_size, Tensors[3] );
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
      }
  }
  //
  //
  // 
  template< typename T,int D > void
  Alps::LayerTensors< T, D >::replace( const std::array< std::size_t, D >  Tensor_size,
				       std::array< std::vector< T >, 4 >   Tensors )
  {
    try
      {
	//
	// Save the previous set of neurons
	// ToDo: implement de move sementic in AlpsImage
	// ToDo: it cost a lot of memory previous_epoque_tensors_.push_back( /*std::move(*/tensors_/*)*/ );
	// Load new tensors
	tensors_[0] = Alps::Image< double, D >( Tensor_size, Tensors[0] );
	tensors_[1] = Alps::Image< double, D >( Tensor_size, Tensors[1] );
	tensors_[2] = Alps::Image< double, D >( Tensor_size, Tensors[2] );
	tensors_[3] = Alps::Image< double, D >( Tensor_size, Tensors[3] );
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
      }
  }
}
#endif
