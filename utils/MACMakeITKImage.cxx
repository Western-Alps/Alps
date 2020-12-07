#include "MACMakeITKImage.h"

//
//
//
MAC::MACMakeITKImage::MACMakeITKImage( const long unsigned int Dimension,
				       const std::string&      Image_name,
				       const Reader<3>::Pointer Dim_template_image ):
  D_{ Dimension }, image_name_{ Image_name }
{
  //
  //
  images_.resize( Dimension );
  //
  // Take the dimension of the first subject image:
  //Reader<3>::Pointer image_reader_ = Dim_template_image;
  image_reader_ = Dim_template_image;

  //
  // Create the 3D image of measures
  //

  //
  //
  ImageType<3>::RegionType region;
  ImageType<3>::IndexType  start = { 0, 0, 0 };
  //
  ImageType<3>::Pointer  raw_subject_image_ptr = image_reader_->GetOutput();
  ImageType<3>::SizeType size = raw_subject_image_ptr->GetLargestPossibleRegion().GetSize();
  //
  region.SetSize( size );
  region.SetIndex( start );
  //
  for ( auto& image : images_ )
    {
      image = ImageType<3>::New();
      image->SetRegions( region );
      image->Allocate();
      image->FillBuffer( 0.0 );
    }
}
//
//
//
void
MAC::MACMakeITKImage::set_val( const std::size_t Image_number, 
			       const MaskType<3>::IndexType Idx, 
			       const double Val )
{
  images_[ Image_number ]->SetPixel( Idx, Val );
}
//
//
//
void
MAC::MACMakeITKImage::write() 
{
  //
  // 
  using Iterator3D = itk::ImageRegionConstIterator< ImageType<3> >;
  using Iterator4D = itk::ImageRegionIterator< ImageType<4> >;
  using FilterType = itk::ChangeInformationImageFilter< ImageType<4> >;

  //
  // Create the 4D image of measures
  //

  //
  // Set the measurment 4D image
  ImageType<4>::Pointer records = ImageType<4>::New();
  //
  ImageType<4>::RegionType region;
  ImageType<4>::IndexType  start = { 0, 0, 0, 0 };
  //
  // Take the dimension of the first subject image:
  Reader<3>::Pointer Sub_image_reader = image_reader_;
  //
  ImageType<3>::Pointer  raw_subject_image_ptr = Sub_image_reader->GetOutput();
  ImageType<3>::SizeType size = raw_subject_image_ptr->GetLargestPossibleRegion().GetSize();
  ImageType<4>::SizeType size_4D{ size[0], size[1], size[2], D_ };

  
  //
  region.SetSize( size_4D );
  region.SetIndex( start );
  //
  records->SetRegions( region );
  records->Allocate();
  //
  // ITK orientation, most likely does not match our orientation
  // We have to reset the orientation
  // Origin
  ImageType<3>::PointType orig_3d = raw_subject_image_ptr->GetOrigin();
  ImageType<4>::PointType origin;
  origin[0] = orig_3d[0]; origin[1] = orig_3d[1]; origin[2] = orig_3d[2]; origin[3] = 0.;
  // Spacing 
  ImageType<3>::SpacingType spacing_3d = raw_subject_image_ptr->GetSpacing();
  ImageType<4>::SpacingType spacing;
  spacing[0] = spacing_3d[0]; spacing[1] = spacing_3d[1]; spacing[2] = spacing_3d[2]; spacing[3] = 1.;
  // Direction
  ImageType<3>::DirectionType direction_3d = raw_subject_image_ptr->GetDirection();
  ImageType<4>::DirectionType direction;
  direction[0][0] = direction_3d[0][0]; direction[0][1] = direction_3d[0][1]; direction[0][2] = direction_3d[0][2]; 
  direction[1][0] = direction_3d[1][0]; direction[1][1] = direction_3d[1][1]; direction[1][2] = direction_3d[1][2]; 
  direction[2][0] = direction_3d[2][0]; direction[2][1] = direction_3d[2][1]; direction[2][2] = direction_3d[2][2];
  direction[3][3] = 1.; // 
  //

  //
  // image filter
  FilterType::Pointer images_filter;
  images_filter = FilterType::New();
  //
  images_filter->SetOutputSpacing( spacing );
  images_filter->ChangeSpacingOn();
  images_filter->SetOutputOrigin( origin );
  images_filter->ChangeOriginOn();
  images_filter->SetOutputDirection( direction );
  images_filter->ChangeDirectionOn();
  //
  //
  Iterator4D it4( records, records->GetBufferedRegion() );
  it4.GoToBegin();
  //
  for ( auto image : images_ )
    {
      ImageType<3>::RegionType region = image->GetBufferedRegion();
      Iterator3D it3( image, region );
      it3.GoToBegin();
      while( !it3.IsAtEnd() )
	{
	  it4.Set( it3.Get() );
	  ++it3; ++it4;
	}
    }
  //
  images_filter->SetInput( records );
  
  //
  // Writer
  std::cout << "Writing the 4d output image for "<< image_name_ <<  std::endl;
  //
  itk::NiftiImageIO::Pointer nifti_io = itk::NiftiImageIO::New();
  //
  itk::ImageFileWriter< ImageType<4> >::Pointer writer = itk::ImageFileWriter< ImageType<4> >::New();
  writer->SetFileName( image_name_ );
  writer->SetInput( images_filter->GetOutput() );
  writer->SetImageIO( nifti_io );
  writer->Update();
}
