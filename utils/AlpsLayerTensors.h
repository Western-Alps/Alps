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
  enum class TensorOrder1
    { 
     UNKNOWN    = -1,
     ACTIVATION =  0,
     DERIVATIVE =  1,
     ERROR      =  2,
     WERROR     =  3  // WEIGHTED_ERROR
    };

  /*! \class LayerTensors
   *
   * \brief class LayerTensors records the Layer flatten images
   * (tensor order 1). It holds for an image m:
   * [0] Activation
   * [1] Derivative of the activation
   * [2] error from the layer
   * [3] weighted error from the layer
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
    virtual ~LayerTensors(){};

    
    //
    // Accessors
    //
    // Get size of the tensor
    virtual const std::vector< std::size_t >    get_tensor_size() const                       override
    { return tensors_[0].get_tensor_size();};
    // Get the tensor
    virtual const std::vector< Type >&          get_tensor() const                            override
    { };
    // Update the tensor
    virtual std::vector< Type >&                update_tensor()                               override 
    { };
//    // Set size of the tensor
//    virtual void                                set_tensor_size( std::vector< std::size_t > ) override{};
//    // Set the tensor
//    virtual void                                set_tensor( std::shared_ptr< Type > )         override{};
    //
    // Access the images directly
    const Alps::Image< Type, Dim >&             get_image( Alps::TensorOrder1 ) const;

    
    //
    // Functions
    //
    // Save the tensor values (e.g. weights)
    virtual void                                save_tensor() const                           override{};
    // Load the tensor values (e.g. weights)
    virtual void                                load_tensor( const std::string )              override{};
    //
    //
    // Implementation of [] operator.  This function must return a 
    // reference as array element can be put on left side 
    std::vector< Type >&                        operator[]( Alps::TensorOrder1 Idx ); 
    //
    // Implementation of () operator.  This function must return a 
    // reference as array element of the Hadamard product between two tensors
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
  Alps::LayerTensors< T, D >::LayerTensors( const std::vector< std::size_t >     Tensor_size,
					    std::array< std::vector< T >, 4 > Tensors )
  {
    try
      {
	//
	// Load the modalities into the container
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
					    std::array< std::vector< T >, 4 > Tensors )
  {
    try
      {
	//
	// Load the modalities into the container
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
  template< typename T,int D > const Alps::Image< T, D >&
  Alps::LayerTensors< T, D >::get_image( Alps::TensorOrder1 Idx ) const 
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
	// ToDo: Fix the problem
//	double b = tensors_[ static_cast< int >( Idx2 ) ].get_tensor().get()[s];
//	double a = 0.;
//	if ( tensors_[ static_cast< int >( Idx1 ) ].get_tensor() )
//	  a = tensors_[ static_cast< int >( Idx1 ) ].get_tensor().get()[s];
//	std::cout
//	  << "img_size["<<s<<"] / "<<img_size 
//	  << " tensors_[Idx1] = " << a
//	  << " tensors_[Idx2] = " << b
//	  << std::endl;
	hadamard[s] =
	  tensors_[ static_cast< int >( Idx1 ) ].get_tensor()[s] +
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
	// Save the previous set of neurons
	// ToDo: implement de move sementic in AlpsImage
	previous_epoque_tensors_.push_back( /*std::move(*/tensors_/*)*/ );
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
	previous_epoque_tensors_.push_back( /*std::move(*/tensors_/*)*/ );
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
