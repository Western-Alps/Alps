#include "Subject.h"
//
//
//
void
MAC::Subject::write() const
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
//
void
MAC::Subject::write_clone() const
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
      std::string name = "subject_clone_" + std::to_string(mod) + ".nii.gz";
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
//
void
MAC::Subject::add_modality( const std::string Mod_name )
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
//
//
void
MAC::Subject::add_modality_target( const std::string Mod_target_name )
{
  if ( file_exists(Mod_target_name) )
    {
      std::cout << Mod_target_name << std::endl;
      //
      // load the image ITK pointer
      auto image_ptr = itk::ImageIOFactory::CreateImageIO( Mod_target_name.c_str(),
							   itk::ImageIOFactory::ReadMode );
      image_ptr->SetFileName( Mod_target_name );
      image_ptr->ReadImageInformation();
      // Read the ITK image
      Reader3D::Pointer img_ptr = Reader3D::New();
      img_ptr->SetFileName( image_ptr->GetFileName() );
      img_ptr->Update();
      //
      modality_targets_ITK_images_.push_back( img_ptr->GetOutput() );
    }
  else
    {
      std::string mess = "Image (";
      mess +=  Mod_target_name + ") does not exists.";
      throw MAC::MACException( __FILE__, __LINE__,
			       mess.c_str(),
			       ITK_LOCATION );
    }
};
//
//
//
void
MAC::Subject::update()
{
  clone_modalities_images_.resize( modalities_ITK_images_.size() );
  //
  for ( int img = 0 ; img < static_cast< int >( modalities_ITK_images_.size() ) ; img++ )
    clone_modalities_images_[img] = modalities_ITK_images_[img];
};
//
//
// 
void
MAC::Subject::update( const std::vector< Image3DType::Pointer > New_images )
{
  clone_modalities_images_.resize( New_images.size() );
  //
  for ( int img = 0 ; img < static_cast< int >( New_images.size() ) ; img++ )
    clone_modalities_images_[img] = New_images[img];
};
