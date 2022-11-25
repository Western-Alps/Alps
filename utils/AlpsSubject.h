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
#ifndef ALPSSUBJECT_H
#define ALPSSUBJECT_H
//
//
//
#include <iostream> 
#include <memory>
#include <filesystem>
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
#include "AlpsLoadDataSet.h"
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
    // Get epoque
    virtual const std::size_t                         get_epoque() const                   override
    { return epoque_;};
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
    std::vector< std::shared_ptr< Alps::LayerTensors< double, Dim > > >& get_layer( const std::string );
    // set images energy for the epoque
    void                                              set_energy( const double E ) 
    { energy_.push_back(E);};


    //
    // functions
    //
    // Update the subject information
    virtual void                                      update()                             override
    {epoque_++;};
    // Visualization of the processed image
    virtual void                                      visualization()                      override;
    //
    //
    // Add a modality
    void                                              add_modalities( const std::string );
    // Check the modalities
    const bool                                        check_modalities() const 
    { return (number_modalities_ == modalities_.size() ? true : false);};
    // Add a dense layer
    void                                              add_layer( const std::string,
								 const std::vector< std::size_t >,
								 std::array< std::vector< double >, 4 > );
    // Add a convolutional layer
    void                                              add_layer( const std::string,
								 const int,
								 const std::array< std::size_t, Dim >,
								 std::array< std::vector< double >, 4 > );
    // Add target in the classification study case
    void                                              add_target( const std::size_t, const std::size_t );
    // Add target using images
    void                                              add_target( const std::string );

    //
    // Private function
  private:
    void                                              fcl_conditioning();
    
  private:
    //
    // Number of epoques
    std::size_t                                                  epoque_{0};
    // energy for the set of input images for each epoque
    std::vector< double >                                        energy_;
    // Vector of modalities 
    std::vector< Alps::LayerTensors< double, Dim > >             modalities_;
    // Vector of modalities 
    std::map< std::string,
	      std::vector< Alps::LayerTensors< double, Dim > > > layer_modalities_;
    // Vector of modalities 
    std::map< std::string,
	      std::vector< std::shared_ptr< Alps::LayerTensors< double, Dim > > > > layer_modalities_ptr_;
    // Vector of modalities references
    std::map< std::string,
	      std::vector< std::reference_wrapper< Alps::LayerTensors< double, Dim > > > > layer_modalities_ref_;
    // Vector of output names
    std::map< std::string,
	      std::vector< std::string > >                       layer_IO_;
    // Vector of modalities 
    Alps::Image< double, Dim >                                   target_;
    //
    // Subject information
    int                                                          subject_number_{0};
    // number of modalities
    std::size_t                                                  number_modalities_{0};
    //
    // Output directory
    std::string                                                  output_directory_{""};
  };
  //
  //
  // Constructor
  template< int D >
  Alps::Subject< D >::Subject( const int         SubNumber,
			       const std::size_t NumModalities ):
    subject_number_{SubNumber}, number_modalities_{NumModalities}
  {
    layer_modalities_["__input_layer__"] = std::vector< Alps::LayerTensors< double, D > >();
    layer_modalities_ptr_["__input_layer__"] = std::vector< std::shared_ptr< Alps::LayerTensors< double, D > > >();
    //
    output_directory_ = Alps::LoadDataSet::instance()->get_data()["mountain"]["IO"]["output_dir"];
  }
  //
  //
  // 
  template< int D > void
  Alps::Subject< D >::visualization()
  {
    try
      {
	for ( std::size_t m = 0 ; m < number_modalities_ ; m++ )
	  {
	    //
	    // Input characteristics
	    // load the image ITK pointer
	    auto image_ptr = itk::ImageIOFactory::CreateImageIO( layer_IO_["__input__"][m].c_str(),
								 itk::CommonEnums::IOFileMode::ReadMode );
	    image_ptr->SetFileName( layer_IO_["__input__"][m] );
	    image_ptr->ReadImageInformation();
	    // Check the dimensions complies
	    if ( image_ptr->GetNumberOfDimensions() != D )
	      throw MAC::MACException( __FILE__, __LINE__,
				       "The dimensions of the image and instanciation are different.",
				       ITK_LOCATION );
	    // Read the ITK image
	    typename Reader< D >::Pointer img_ptr = Reader< D >::New();
	    img_ptr->SetFileName( image_ptr->GetFileName() );
	    img_ptr->Update();
	    //
	    // Output characteristics
	    std::string out_name = layer_IO_["__output__"][m] + "_" + std::to_string( epoque_ ) + ".nii.gz";
	    //
	    //
	    layer_modalities_["__output_layer__"][m].get_image( Alps::TensorOrder1::ACTIVATION ).visualization( out_name,
														img_ptr );
	    layer_modalities_ptr_["__output_layer__"][m]->get_image( Alps::TensorOrder1::ACTIVATION ).visualization( out_name,
														     img_ptr );
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
  Alps::Subject< D >::add_modalities( const std::string Modality )
  {
    try
      {
	if ( Alps::file_exists(Modality) )
	  {
	    //
	    // Load the modalities into the container
	    layer_modalities_["__input_layer__"].push_back( Alps::LayerTensors< double, D >(Modality) );
	    layer_modalities_ptr_["__input_layer__"].push_back( std::make_shared< Alps::LayerTensors< double, D > >(Modality) );
	    modalities_.push_back( Alps::LayerTensors< double, D >(Modality) );
	    //
	    // Create the output modality
	    std::string output_modality = std::filesystem::path( Modality ).stem();
	    if ( output_modality.ends_with(".nii") )
	      output_modality = output_modality.replace( output_modality.end() - 4,
							 output_modality.end(), "" );
	    layer_IO_["__output__"].push_back( std::filesystem::path(output_directory_) / output_modality );
	    //
	    layer_IO_["__input__"].push_back( Modality );
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
  template< int D > std::vector< std::shared_ptr< Alps::LayerTensors< double, D > > >&
  Alps::Subject< D >::get_layer( const std::string Layer_name ) 
  {
    //
    // Check the layer exist
    auto layer = layer_modalities_.find( Layer_name );
    //
    try
      {
	if ( layer == layer_modalities_.end() )
//	  {
//	    if ( (layer->second).size() == 0 )
//	      {
//		if ( Layer_name == "__input_layer__" )
//		  fcl_conditioning();
//		else
//		  {
//		    std::string mess = "Layer " + Layer_name + " is empty.";
//		    throw MAC::MACException( __FILE__, __LINE__,
//					     mess.c_str(),
//					     ITK_LOCATION );
//		  }
//	      }
//	  }
//	else
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
    // If the map of vector references is empty,
    // we are going through the map to create reference vectors
    if ( layer_modalities_ref_.empty() )
      {
	//
	auto iter = layer_modalities_.begin();
	//
	while ( iter != layer_modalities_.end() )
	  {
	    layer_modalities_ref_[ iter->first ] =
	      std::vector< std::reference_wrapper< Alps::LayerTensors< double, D > > > ( iter->second.begin(),
											 iter->second.end() );
	    ++iter;
	  }
      }
    // Check if it should have the key
    if ( layer_modalities_.find( Layer_name ) != layer_modalities_.end() &&
	 layer_modalities_ref_.find( Layer_name ) == layer_modalities_ref_.end() )
      {
	layer_modalities_ref_[ Layer_name ] =
	      std::vector< std::reference_wrapper< Alps::LayerTensors< double, D > > > ( layer_modalities_[ Layer_name ].begin(),
											 layer_modalities_[ Layer_name ].end() );
      }
    
    //
    //
    return layer_modalities_ptr_[Layer_name];
  }
  //
  //
  // 
  template< int D > void
  Alps::Subject< D >::add_layer( const std::string                      Layer_name,
				 const std::vector<std::size_t>         Layer_size,
				 std::array< std::vector< double >, 4 > Tensors_activation )
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
	    layer_modalities_ptr_[ Layer_name ] = std::vector< std::shared_ptr< Alps::LayerTensors< double, D > > >();
	    layer_modalities_ptr_[ Layer_name ].push_back( std::make_shared< Alps::LayerTensors< double, D > >(Layer_size,
													       Tensors_activation) );
	  }
	else
	  {
	    layer_modalities_[ Layer_name ][0].replace( Layer_size,
							Tensors_activation );
	    layer_modalities_ptr_[ Layer_name ][0]->replace( Layer_size,
							     Tensors_activation );
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
  Alps::Subject< D >::add_layer( const std::string                      Layer_name,
				 const int                              Kernel,
				 const std::array< std::size_t, D >     Layer_size,
				 std::array< std::vector< double >, 4 > Tensors_activation )
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
	    layer_modalities_ptr_[ Layer_name ] = std::vector< std::shared_ptr< Alps::LayerTensors< double, D > > >();
	    layer_modalities_ptr_[ Layer_name ].push_back( std::make_shared< Alps::LayerTensors< double, D > >(Layer_size,
													   Tensors_activation) );
	  }
	else
	  {
	    if ( layer_modalities_[ Layer_name ].size() < static_cast< std::size_t >(Kernel + 1) )
	      {
		layer_modalities_[ Layer_name ].push_back( Alps::LayerTensors< double, D >(Layer_size,
											   Tensors_activation) );
		layer_modalities_ptr_[ Layer_name ].push_back( std::make_shared< Alps::LayerTensors< double, D > >(Layer_size,
														   Tensors_activation) );
		//
		if ( layer_modalities_[ Layer_name ].size() != static_cast< std::size_t >(Kernel + 1) )
		  {
		    std::string mess = "Synchronization issue between the number of features recorded and the current feature.";
		    throw MAC::MACException( __FILE__, __LINE__,
					     mess.c_str(),
					     ITK_LOCATION );
		  }
	      }
	    else
	      {
		layer_modalities_[ Layer_name ][Kernel].replace( Layer_size,
								 Tensors_activation );
		layer_modalities_ptr_[ Layer_name ][Kernel]->replace( Layer_size,
								      Tensors_activation );
	      }
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
  Alps::Subject< D >::add_target( const std::size_t Label,
				  const std::size_t Universe)
  {
    try
      {
	if ( Label < Universe + 1 )
	  {
	    // initialize the label to zero
	    target_ = Alps::Image< double, D >( std::vector< std::size_t >(/*tensor order*/ 1, Universe ),
						std::vector< double >( Universe, 0. ) );
	    // Set the label value
	    target_.update_tensor()[Label] = 1.;
	  }
	else
	  {
	    std::string mess = "The label (";
	    mess            += std::to_string(Label) + ") can't be bigger than the Unverse size (";
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
  Alps::Subject< D >::add_target( const std::string Image )
  {
    try
      {
	if ( Alps::file_exists(Image) )
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
				       "The dimensions of the target image and instanciation are different.",
				       ITK_LOCATION );
	    //
	    // Read the ITK image
	    typename Reader< D >::Pointer img_ptr = Reader< D >::New();
	    img_ptr->SetFileName( image_ptr->GetFileName() );
	    img_ptr->Update();
	    //
	    // Load the modalities into the container
	    target_ = Alps::Image< double, D >( img_ptr );
	  }
	else
	  {
	    std::string mess = "Image (";
	    mess            += Image + ") does not exists.";
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
	    // activation function, in this case input
	    std::vector< double > z( size, 0. );
	    // Derivative of the activation function
	    std::vector< double > dz( size, 0. );
	    // Error back propagated in building the gradient
	    std::vector< double > error( size, 0. );
	    // Weighted error back propagated in building the gradient
	    std::vector< double > werr( size, 0. );
	    //
	    std::size_t idx = 0;
	    for ( auto mod = modalities_.begin() ;
		  mod != modalities_.end() ; mod++  )
	      {
		std::size_t sub_idx = (*mod).get_tensor_size()[0];
		for ( std::size_t i = 0 ; i < sub_idx ; i++)
		  z[idx++] = (*mod)[Alps::TensorOrder1::ACTIVATION][i];
	      }
	    //
	    if ( idx != size )
	      {
		std::cout
		  << "idx = " << idx
		  << " && size = " << size
		  << std::endl;
		std::string mess = "There is missmatch between the size expected (" + std::to_string(idx);
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
									       {z, dz, error, werr}) );
	    layer_modalities_ptr_[name].push_back( std::make_shared< Alps::LayerTensors< double, D > >(layer_size,
												       {z, dz, error, werr}) );
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
