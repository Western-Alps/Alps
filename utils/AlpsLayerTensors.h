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
     UNKNOWN = -1,
     NEURONS =  0,
     ERRORS  =  1,
    };

  /*! \class LayerTensors
   *
   * \brief class LayerTensors records the Layer flatten images
   * (tensor order 1). And memorized the tensors from 
   * previous epoques.
   *
   */
  template< typename Type, int Dim >
  class LayerTensors : public Alps::Tensor< Type, 1 >
  {
  public:
    /** Constructor */
    LayerTensors( const std::string );
    /** Constructor */
    LayerTensors( const std::vector< std::size_t >, std::shared_ptr< double > );
    /* Destructor */
    virtual ~LayerTensors(){};

    
    //
    // Accessors
    //
    // Get size of the tensor
    virtual const std::vector< std::size_t >    get_tensor_size() const                       override
    { return tensors_[0].get()->get_tensor_size();};
    // Get the tensor
    virtual std::shared_ptr< Type >             get_tensor() const                            override
    { return nullptr;};
    // Set size of the tensor
    virtual void                                set_tensor_size( std::vector< std::size_t > ) override{};
    // Set the tensor
    virtual void                                set_tensor( std::shared_ptr< Type > )         override{};

    
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
    void                                        replace( const std::vector<std::size_t>,
							 std::shared_ptr< double > ){};  
    
  private:
    // (Z,error, )
    std::array< std::shared_ptr< Alps::Image< Type, Dim > >, 2 > tensors_{nullptr,nullptr};
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
	tensors_[ 0 ] = std::make_shared< Alps::Image< double, D > >( Alps::Image< double, D >(img_ptr) );
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
  Alps::LayerTensors< T, D >::LayerTensors( const std::vector< std::size_t > Tensor_size,
					    std::shared_ptr< double >        Tensor )
  {
    try
      {
	//
	// Load the modalities into the container
	tensors_[0] = std::make_shared< Alps::Image< double, D > >( Alps::Image< double, D >(Tensor_size,
											     Tensor) );
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
      }
  }
  //
  //
  // Operator []
  template< typename T,int D > T*
  Alps::LayerTensors< T, D >::operator[]( Alps::TensorOrder1 Idx ) 
  {
    try
      {
	if ( static_cast< int >( Idx ) > 2 ||
	     static_cast< int >( Idx ) < 0 )
	  throw MAC::MACException( __FILE__, __LINE__,
				   "Indexing not implemented yet.",
				   ITK_LOCATION );
	//
	//
	return tensors_[ static_cast< int >( Idx ) ]->get_tensor().get(); 
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
      }
  }
}
#endif