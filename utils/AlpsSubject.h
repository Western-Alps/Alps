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
#include "AlpsLayerTensors.h"
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
  template< int Dim >
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
    virtual std::shared_ptr< Alps::Mountain >         get_mountain()                       override
    { return nullptr;};
    //
    //
    // Subject information
    const int                                         get_subject_number() const
    {return subject_number_;};
    // number of modalities
    const int                                         get_number_modalities() const
    {return number_modalities_;};
    // Return the size of the layers
    std::vector<std::size_t>                          get_layer_size();
    // Get layer modality z
    std::vector< Alps::LayerTensors< double, Dim > >& get_layer( const std::string Layer )
    { return modalities_[Layer]; };


    //
    // functions
    //
    // Update the subject information
    virtual void                                      update()                             override{};
    //
    //
    // Add a modality
    void                                              add_modalities( const std::string );
    // Check the modalities
    const bool                                        check_modalities( const std::string Layer ) 
      { return (number_modalities_ == modalities_[Layer].size() ? true : false);};
    // Add a layer
    void                                              add_layer( const std::string,
								 const std::vector<std::size_t>,
								 std::shared_ptr< double >  );

    
  private:
    //
    // Vector of modalities 
    std::map< std::string, std::vector< Alps::LayerTensors< double, Dim > > > modalities_;
    //
    // Subject information
    int         subject_number_{0};
    // number of modalities
    std::size_t number_modalities_{0};
  };
  //
  //
  // Constructor
  template< /*class F,*/ int D >
  Alps::Subject<D>::Subject( const int         SubNumber,
			     const std::size_t NumModalities ):
    subject_number_{SubNumber}, number_modalities_{NumModalities}
  {
    modalities_["__input_layer__"] = std::vector< Alps::LayerTensors< double, D > >();
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
	    // Load the modalities into the container
	    modalities_["__input_layer__"].push_back( Alps::LayerTensors< double, D >(Modality) );
	  }
	else
	  {
	    std::string mess = "Image (";
	    mess            += Modality + ") does not exists.";
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
  template< /*class F,*/ int D > std::vector<std::size_t>
  Alps::Subject< D >::get_layer_size()
  {
    try
      {
	std::vector<std::size_t> layer_size;
	for ( int img_in = 0 ; img_in < number_modalities_ ; img_in++ )
	  layer_size.push_back( modalities_["__input_layer__"][img_in].get_tensor_size()[0] );
	//
	return layer_size;
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
      }
  }
  //
  //
  // 
  template< /*class F,*/ int D > void
  Alps::Subject< D >::add_layer( const std::string         Layer_name,
				 const std::vector<std::size_t>    Layer_size,
				 std::shared_ptr< double > Tensor_activation )
  {
    try
      {
	//
	// Check the layer exist
	auto layer = modalities_.find( Layer_name );
	//
	if ( layer == modalities_.end() )
	  {
	    modalities_[ Layer_name ] = std::vector< Alps::LayerTensors< double, D > >();
	    modalities_[ Layer_name ].push_back( Alps::LayerTensors< double, D >(Layer_size,
										 Tensor_activation) );
	  }
	else
	  modalities_[ Layer_name ][0].replace( Layer_size,
						Tensor_activation );
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
      }
  }
}
#endif
