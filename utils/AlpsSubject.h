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
    // get images energy
    virtual const double                              get_energy() const                   override
    { return energy_.back();};
    //
    //
    // Subject information
    const int                                         get_subject_number() const
    { return subject_number_;};
    // get images energy
    const std::vector< double >&                      get_energies() const
    { return energy_;};
    // taget tensor
    const Alps::Image< double, Dim >&                 get_target() const
    { return target_;};
    // number of modalities
    const int                                         get_number_modalities() const
    { return number_modalities_;};
    // Return the size of the layers
    std::vector<std::size_t>                          get_layer_size( const std::string );
    // Get layer modality z
    std::vector< Alps::LayerTensors< double, Dim > >& get_layer( const std::string );
    // set images energy
    void                                              set_energy( const double E ) 
    { energy_.push_back(E);};


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
    const bool                                        check_modalities() const 
    { return (number_modalities_ == modalities_.size() ? true : false);};
    // Add a layer
    void                                              add_layer( const std::string,
								 const std::vector<std::size_t>,
								 std::tuple< std::shared_ptr< double >,
								              std::shared_ptr< double >,
								              std::shared_ptr< double >,
								              std::shared_ptr< double > > );
    // Add target in the classification study case
    void                                              add_target( const std::size_t, const std::size_t );

    //
    // Private function
  private:
    void                                              fcl_conditioning();
    
  private:
    //
    // energy for the set of input images for each epoque
    std::vector< double >                                        energy_;
    // Vector of modalities 
    std::vector< Alps::LayerTensors< double, Dim > >             modalities_;
    // Vector of modalities 
    std::map< std::string,
	      std::vector< Alps::LayerTensors< double, Dim > > > layer_modalities_;
    // Vector of modalities 
    Alps::Image< double, Dim >                                   target_;
    //
    // Subject information
    int                                                          subject_number_{0};
    // number of modalities
    std::size_t                                                  number_modalities_{0};
  };
  //
  //
  // Constructor
  template< /*class F,*/ int D >
  Alps::Subject< D >::Subject( const int         SubNumber,
			       const std::size_t NumModalities ):
    subject_number_{SubNumber}, number_modalities_{NumModalities}
  {
    layer_modalities_["__input_layer__"] = std::vector< Alps::LayerTensors< double, D > >();
  }
  //
  //
  // 
  template< int D > void
  Alps::Subject< D >::add_modalities( const std::string Modality )
  {
    try
      {
	if ( Alps::file_exists(Modality) )
	  {
	    //
	    // Load the modalities into the container
	    //layer_modalities_["__input_layer__"].push_back( Alps::LayerTensors< double, D >(Modality) );
	    modalities_.push_back( Alps::LayerTensors< double, D >(Modality) );
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
  template< int D > std::vector<std::size_t>
  Alps::Subject< D >::get_layer_size( const std::string Layer_name ) 
  {
    //
    // Check the layer exist
    auto layer = layer_modalities_.find( Layer_name );
    //
    try
      {
	if ( layer != layer_modalities_.end() )
	  {
	    if ( (layer->second).size() == 0 )
	      {
		if ( Layer_name == "__input_layer__" )
		  fcl_conditioning();
		else
		  {
		    std::string mess = "Layer " + Layer_name + " is empty.";
		    throw MAC::MACException( __FILE__, __LINE__,
					     mess.c_str(),
					     ITK_LOCATION );
		  }
	      }
	  }
	else
	  {
	    std::string mess = "Layer " + Layer_name + " is unknown.";
	    throw MAC::MACException( __FILE__, __LINE__,
				     mess.c_str(),
				     ITK_LOCATION );
	  }
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
      }
    //
    //
    return (layer->second)[0].get_tensor_size();
  }
  //
  //
  //
  template< int D > std::vector< Alps::LayerTensors< double, D > >&
  Alps::Subject< D >::get_layer( const std::string Layer_name ) 
  {
    //
    // Check the layer exist
    auto layer = layer_modalities_.find( Layer_name );
    //
    try
      {
	if ( layer != layer_modalities_.end() )
	  {
	    if ( (layer->second).size() == 0 )
	      {
		if ( Layer_name == "__input_layer__" )
		  fcl_conditioning();
		else
		  {
		    std::string mess = "Layer " + Layer_name + " is empty.";
		    throw MAC::MACException( __FILE__, __LINE__,
					     mess.c_str(),
					     ITK_LOCATION );
		  }
	      }
	  }
	else
	  {
	    std::string mess = "Layer " + Layer_name + " is unknown.";
	    throw MAC::MACException( __FILE__, __LINE__,
				     mess.c_str(),
				     ITK_LOCATION );
	  }
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
      }
    //
    //
    return layer_modalities_[Layer_name];
  }
  //
  //
  // 
  template< int D > void
  Alps::Subject< D >::add_layer( const std::string                       Layer_name,
				 const std::vector<std::size_t>          Layer_size,
				 std::tuple< std::shared_ptr< double >,
				             std::shared_ptr< double >,
				             std::shared_ptr< double >,
				             std::shared_ptr< double > > Tensors_activation )
  {
    try
      {
	//
	// Check the layer exist
	auto layer = layer_modalities_.find( Layer_name );
	//
	if ( layer == layer_modalities_.end() )
	  {
	    layer_modalities_[ Layer_name ] = std::vector< Alps::LayerTensors< double, D > >();
	    layer_modalities_[ Layer_name ].push_back( Alps::LayerTensors< double, D >(Layer_size,
										       Tensors_activation) );
	  }
	else
	  layer_modalities_[ Layer_name ][0].replace( Layer_size,
						      Tensors_activation );
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
      }
  }
  //
  //
  // 
  template< int D > void
  Alps::Subject< D >::add_target( const std::size_t Target,
				  const std::size_t Universe)
  {
    try
      {
	if ( Target < Universe + 1 )
	  {
	    target_ = Alps::Image< double, D >( std::vector< std::size_t >(/*tensor order*/ 1, Universe ),
						std::shared_ptr< double >(new double[Universe],
									  std::default_delete< double[] >()) );
	    // initialize the target to zero
	    for ( std::size_t i = 0 ; i < Universe ; i++ )
	      (target_.get_tensor().get())[i] = 0.;
	    // Set the target value
	    (target_.get_tensor().get())[Target] = 1.;
	  }
	else
	  {
	    std::string mess = "The target (";
	    mess            += std::to_string(Target) + ") can't be bigger than the Unverse size (";
	    mess            += std::to_string(Universe) + ").";
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
  template< int D > void
  Alps::Subject< D >::fcl_conditioning()
  {
    try
      {
	if ( layer_modalities_.size() > 0 )
	  {
	    //
	    std::vector< std::size_t > layer_size;
	    std::size_t                size = 0;
	    // Get the size
	    for ( auto mod = modalities_.begin() ;
		  mod != modalities_.end() ; mod++  )
	      size += (*mod).get_tensor_size()[0];
	    //
	    // go over the images' tensors
	    std::shared_ptr< double > z  = std::shared_ptr< double >( new  double[ size ],
								      std::default_delete< double[] >() );
	    //
	    std::size_t idx = 0;
	    for ( auto mod = modalities_.begin() ;
		  mod != modalities_.end() ; mod++  )
	      {
		std::size_t sub_idx = (*mod).get_tensor_size()[0];
		for ( std::size_t i = 0 ; i < sub_idx ; i++)
		  z.get()[idx++] = (*mod)[Alps::TensorOrder1::ACTIVATION][i];
	      }
	    //
	    if ( idx != size )
	      {
		std::cout
		  << "idx = " << idx
		  << " && size = " << size
		  << std::endl;
		std::string mess = "There is miss match between the size expected (" + std::to_string(idx);
		mess += ") and the size retrieved ("+ std::to_string(size) +")";
		throw MAC::MACException( __FILE__, __LINE__,
					 mess.c_str(),
					 ITK_LOCATION );
	      }
	    //
	    // Record in the layer_modalities container
	    std::string name = "__input_layer__";
	    layer_size.push_back(size);
	    layer_modalities_[name].push_back( Alps::LayerTensors< double, D >(layer_size,
									       std::make_tuple( z,
												nullptr,
												nullptr,
												nullptr )) );
	  }
	else
	  {
	    std::string mess = "Modalities have not been loaded yet.";
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
}
#endif
