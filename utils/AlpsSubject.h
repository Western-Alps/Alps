#ifndef ALPSSUBJECT_H
#define ALPSSUBJECT_H
//
//
//
#include <iostream> 
#include <memory>
//
// ITK
//
#include "ITKHeaders.h"
//
// 
//
#include "MACException.h"
#include "AlpsClimber.h"
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
  /*! \class Subject
   *
   * \brief class Subject record the information 
   * of the subject through the processing.
   *
   */
  template< /*class Function,*/ int Dim >
  class Subject : public Alps::Climber
  {
  public:
    //
    /** Constructor */
    explicit Subject( const int,
		      const std::size_t );
    /* Destructor */
    virtual ~Subject(){};

    
    //
    // Accessors
    //
    // Get the observed mountain
    virtual std::shared_ptr< Alps::Mountain >                     get_mountain()                                  override
    { return nullptr;};
    //
    //
    // Subject information
    const int                                                     get_subject_number()    const
    {return subject_number_;};
    // number of modalities
    const int                                                     get_number_modalities() const
    {return number_modalities_;};
    // Return the size of the layers
    std::vector<int>                                              get_layer_size();
    // Get layer modality z
    std::vector< std::shared_ptr< Alps::Image< double, Dim > > >& get_layer_z( const std::string Layer )
    { return modalities_z_[Layer]; };
    // Get layer modality z
    std::vector< std::shared_ptr< Alps::Image< double, Dim > > >& get_layer_eps( const std::string Layer )
    { return modalities_eps_[Layer]; };

    //
    // functions
    //
    // Update the subject information
    virtual void                                                  update()                                    override{};
    //
    //
    // Add a modality
    void                                                          add_modalities( const std::string );
    // Check the modalities
    const bool                                                    check_modalities( const std::string Layer ) 
      { return (number_modalities_ == modalities_z_[Layer].size() ? true : false);};

    
  private:
    //
    // Vector of modalities 
    std::map< std::string, std::vector< std::shared_ptr< Alps::Image< double, Dim > > > >  modalities_z_;
    // Vector of modalities 
    std::map< std::string, std::vector< std::shared_ptr< Alps::Image< double, Dim > > > >  modalities_eps_;
    //
    // Subject information
    int         subject_number_{0};
    // number of modalities
    std::size_t number_modalities_{0};
    //
    // This function is the continuous step function
    /*Function                                         activation_function_;*/
  };
  //
  //
  // Constructor
  template< /*class F,*/ int D >
  Alps::Subject<D>::Subject( const int SubNumber,
			     const std::size_t NumModalities ):
    subject_number_{SubNumber}, number_modalities_{NumModalities}
  {
    modalities_z_["__input_layer__"] = std::vector< std::shared_ptr< Alps::Image< double, D > > >();
  }
  //
  //
  // 
  template< /*class F,*/ int D > void
  Alps::Subject< D >::add_modalities( const std::string Modality )
  {
    try
      {
	if ( Alps::file_exists(Modality) )
	  {
	    //
	    // load the image ITK pointer
	    auto image_ptr = itk::ImageIOFactory::CreateImageIO( Modality.c_str(),
								 itk::CommonEnums::IOFileMode::ReadMode );
	    image_ptr->SetFileName( Modality );
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
	    modalities_z_["__input_layer__"].push_back( std::make_shared< Alps::Image< double, D > >(Alps::Image< double, D >(img_ptr)) );
	  }
	else
	  {
	    std::string mess = "Image (";
	    mess += Modality + ") does not exists.";
	    throw MAC::MACException( __FILE__, __LINE__,
				     mess.c_str(),
				     ITK_LOCATION );
	  }
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
      }
  }
  //
  //
  //
  template< /*class F,*/ int D > std::vector<int>
  Alps::Subject< D >::get_layer_size()
  {
    try
      {
	std::vector<int> layer_size;
	for ( int img_in = 0 ; img_in < number_modalities_ ; img_in++ )
	  layer_size.push_back( std::dynamic_pointer_cast< Alps::Image< double, D > >(modalities_z_["__input_layer__"][img_in])->get_tensor_size()[0] );
	//
	return layer_size;
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
      }
  }
}
#endif
