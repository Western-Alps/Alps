#ifndef SUBJECT_H
#define SUBJECT_H
//
//
//
#include <iostream>
#include <fstream>
#include <math.h>  
#include <map>
#include <list>
//
// ITK
//
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>
#include <itkNiftiImageIO.h>
#include <itkOrientImageFilter.h>
#include <itkSpatialOrientation.h>
// Some typedef
using Image3DType = itk::Image< double, 3 >;
using Reader3D    = itk::ImageFileReader< Image3DType >;
using MaskType    = itk::Image< unsigned char, 3 >;
//
//
//
#include "MACException.h"
//
//
//
namespace MAC
{
  inline bool file_exists ( const std::string& name )
  {
    std::ifstream f( name.c_str() );
    return f.good();
  }

  /** \class Subject
   *
   * \brief 
   * Each subject loads a pointer for all modalities
   * 
   */
  class Subject
    {
      //
      // 
    public:
      /** Constructor. */
      explicit Subject():
      label_{0}{};
    
      /** Destructor */
      virtual ~Subject(){};

      //
      // Accessors
      const std::vector< Image3DType::Pointer >& get_modalities_ITK_images() const
      {
	return modalities_ITK_images_;
      }
      const std::vector< Image3DType::SizeType >& get_modality_images_size() const
      {
	return modality_images_size_;
      }
      const std::vector< Image3DType::Pointer >& get_clone_modalities_images() const
      {
	return clone_modalities_images_;
      }
      std::string get_subject_name() const
	{
	  return name_;
	}
      //
      //
      void set_subject_name( std::string Name )
      {
	name_ = Name;
      }

      //
      // 
      void write() const
      {
	//
	// Check
	int mod = 0;
	for (auto img_ptr : modalities_ITK_images_ )
	  {
	    itk::NiftiImageIO::Pointer nifti_io = itk::NiftiImageIO::New();
	    //
	    itk::ImageFileWriter< Image3DType >::Pointer writer =
	      itk::ImageFileWriter< Image3DType >::New();
	    //
	    std::string name = "sunbject_" + std::to_string(mod) + ".nii.gz";
	    writer->SetFileName( name );
	    writer->SetInput( img_ptr );
	    writer->SetImageIO( nifti_io );
	    writer->Update();
	    //
	    mod++;
	  }
      };
      //
      // 
      void write_clone() const
      {
	//
	// Check
	int mod = 0;
	for (auto img_ptr : clone_modalities_images_ )
	  {
	    itk::NiftiImageIO::Pointer nifti_io = itk::NiftiImageIO::New();
	    //
	    itk::ImageFileWriter< Image3DType >::Pointer writer =
	      itk::ImageFileWriter< Image3DType >::New();
	    //
	    std::string name = "sunbject_clone_" + std::to_string(mod) + ".nii.gz";
	    writer->SetFileName( name );
	    writer->SetInput( img_ptr );
	    writer->SetImageIO( nifti_io );
	    writer->Update();
	    //
	    mod++;
	  }
      };

      //
      // Add modality
      void add_modality( const std::string Mod_name )
      {
	if ( file_exists(Mod_name) )
	  {
	    std::cout << Mod_name << std::endl;
	    //
	    // load the image ITK pointer
	    auto image_ptr = itk::ImageIOFactory::CreateImageIO( Mod_name.c_str(),
								 itk::ImageIOFactory::ReadMode );
	    image_ptr->SetFileName( Mod_name );
	    image_ptr->ReadImageInformation();
	    // Read the ITK image
	    Reader3D::Pointer img_ptr = Reader3D::New();
	    img_ptr->SetFileName( image_ptr->GetFileName() );
	    img_ptr->Update();
	    //
	    modalities_ITK_images_.push_back( img_ptr->GetOutput() );
	    modality_images_size_.push_back( img_ptr->GetOutput()->GetLargestPossibleRegion().GetSize() );
	  }
	else
	  {
	    std::string mess = "Image (";
	    mess +=  Mod_name + ") does not exists.";
	    throw MAC::MACException( __FILE__, __LINE__,
				     mess.c_str(),
				     ITK_LOCATION );
	  }
      };
      //
      // Add label
      void add_label( const int Label )
      {
	label_ = Label;
      };
      //
      // Update the current read image
      void update()
      {
	clone_modalities_images_.resize( modalities_ITK_images_.size() );
	//
	for ( int img = 0 ; img < static_cast< int >( modalities_ITK_images_.size() ) ; img++ )
	  clone_modalities_images_[img] = modalities_ITK_images_[img];
      };
      //
      // Update the current read image
      void update( const std::vector< Image3DType::Pointer > New_images )
      {
	clone_modalities_images_.resize( New_images.size() );
	//
	for ( int img = 0 ; img < static_cast< int >( New_images.size() ) ; img++ )
	  clone_modalities_images_[img] = New_images[img];
      };

    private:
      //
      // private member function
      //

      //
      // Subject parameters
      //

      // subject name
      std::string name_;
      // vector of modalities
      std::vector< Image3DType::Pointer > modalities_ITK_images_;
      // Current read images
      // This set of images will be transfered to next neural network layers
      std::vector< Image3DType::Pointer > clone_modalities_images_;
      // images size
      std::vector< Image3DType::SizeType > modality_images_size_;
      // label
      int label_;
    };
}
#endif
