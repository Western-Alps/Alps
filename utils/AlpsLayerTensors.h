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
		  std::tuple< std::shared_ptr< double >,
		              std::shared_ptr< double >,
		              std::shared_ptr< double >,
		              std::shared_ptr< double > > );
    /* Destructor */
    virtual ~LayerTensors(){};

    
    //
    // Accessors
    //
    // Get size of the tensor
    virtual const std::vector< std::size_t >    get_tensor_size() const                       override
    { return tensors_[0].get_tensor_size();};
    // Get the tensor
    virtual std::shared_ptr< Type >             get_tensor() const                            override
    { return nullptr;};
    // Set size of the tensor
    virtual void                                set_tensor_size( std::vector< std::size_t > ) override{};
    // Set the tensor
    virtual void                                set_tensor( std::shared_ptr< Type > )         override{};
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
    Type*                                       operator[]( Alps::TensorOrder1 Idx ); 
    //
    // Implementation of () operator.  This function must return a 
    // reference as array element of the Hadalart product between two tensors
    std::shared_ptr< Type >                     operator()( Alps::TensorOrder1, Alps::TensorOrder1 ); 
    //
    void                                        replace( const std::vector<std::size_t>,
							  std::tuple< std::shared_ptr< double >,
							              std::shared_ptr< double >,
							              std::shared_ptr< double >,
							              std::shared_ptr< double > >);  
    
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
	tensors_[ 0 ] = Alps::Image< double, D >( img_ptr );
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
  Alps::LayerTensors< T, D >::LayerTensors( const std::vector< std::size_t >         Tensor_size,
					    std::tuple< std::shared_ptr< double >,
					                std::shared_ptr< double >,
					                std::shared_ptr< double >,
					                std::shared_ptr< double > > Tensors )
  {
    try
      {
	//
	// Load the modalities into the container
	tensors_[0] = Alps::Image< double, D >( Tensor_size, std::get< 0 >( Tensors ) );
	tensors_[1] = Alps::Image< double, D >( Tensor_size, std::get< 1 >( Tensors ) );
	tensors_[2] = Alps::Image< double, D >( Tensor_size, std::get< 2 >( Tensors ) );
	tensors_[3] = Alps::Image< double, D >( Tensor_size, std::get< 3 >( Tensors ) );
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
  template< typename T,int D > T*
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
    return tensors_[ static_cast< int >( Idx ) ].get_tensor().get(); 
  }
  //
  //
  // Operator ()
  template< typename T,int D > std::shared_ptr< T >
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
    std::shared_ptr< T > hadamart = std::shared_ptr< T >( new  T[img_size],
							  std::default_delete< T[] >() );
    //
    for ( std::size_t s = 0 ; s < img_size ; s++ )
      hadamart.get()[s] =
	tensors_[ static_cast< int >( Idx1 ) ].get_tensor().get()[s] +
	tensors_[ static_cast< int >( Idx2 ) ].get_tensor().get()[s];
      
    //
    //
    return hadamart;
  }
  //
  //
  // 
  template< typename T,int D > void
  Alps::LayerTensors< T, D >::replace( const std::vector<std::size_t>          Tensor_size,
				       std::tuple< std::shared_ptr< double >,
				                   std::shared_ptr< double >,
				                   std::shared_ptr< double >,
					           std::shared_ptr< double > > Tensors )
  {
    try
      {
	//
	// Save the previous set of neurons
	previous_epoque_tensors_.push_back( tensors_ );
	//
	tensors_    = std::array< Alps::Image< T, D >, 4 >();
	// Load new tensors
	tensors_[0] = Alps::Image< double, D >( Tensor_size, std::get< 0 >( Tensors ) );
	tensors_[1] = Alps::Image< double, D >( Tensor_size, std::get< 1 >( Tensors ) );
	tensors_[2] = Alps::Image< double, D >( Tensor_size, std::get< 2 >( Tensors ) );
	tensors_[3] = Alps::Image< double, D >( Tensor_size, std::get< 3 >( Tensors ) );
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
      }
  }
}
#endif
