#include "testITKFunctions.h"
//
// ITK
//
#include "ITKHeaders.h"

//using ::testing::Return;

ITKFunctionTest::ITKFunctionTest() {
    // Have qux return true by default
    //ON_CALL(m_bar,qux()).WillByDefault(Return(true));
    // Have norf return false by default
    //ON_CALL(m_bar,norf()).WillByDefault(Return(false));
}

ITKFunctionTest::~ITKFunctionTest() {};

void ITKFunctionTest::SetUp() {};

void ITKFunctionTest::TearDown() {};

TEST_F(ITKFunctionTest, SimpleImage) {
  //
  //
  ImageType<2>::RegionType region;
  ImageType<2>::IndexType  start;
  start[0] = 0;
  start[1] = 0;
  //
  ImageType<2>::SizeType size;
  size[0] = 200;
  size[1] = 300;
  //
  region.SetSize( size );
  region.SetIndex( start );
  //
  //
  ImageType<2>::Pointer image = ImageType<2>::New();
  image->SetRegions( region );
  image->Allocate();
  //
  ImageType<2>::IndexType ind;
  ind[0] = 10;
  ind[1] = 10;

  //
  //
  Writer<2>::Pointer writer = Writer<2>::New();
  writer->SetFileName( "test.nii.gz" );
  writer->SetInput(image);

  try
  {
    writer->Update();
  }
  catch (itk::ExceptionObject & error)
  {
    std::cerr << "error" << std::endl;
  }

  EXPECT_EQ( 0, 0 );
}
//
//
// 3D image
TEST_F(ITKFunctionTest, SimpleImage3D) {
  //
  //
  using Image = itk::Image< double, 3 >;
  //
  ImageType<3>::RegionType region;
  ImageType<3>::IndexType  start;
  start = {0,0,0};
  //
  ImageType<3>::SizeType size;
  size = {4,4,4}; 
  //
  region.SetSize( size );
  region.SetIndex( start );
  //
  //
  ImageType<3>::Pointer image = ImageType<3>::New();
  image->SetRegions( region );
  image->Allocate();
  //
  //
  itk::ImageRegionIterator< Image > imageIterator( image, region );
  //
  double val = 0;
  while( !imageIterator.IsAtEnd() )
    {
      ImageType< 3 >::IndexType idx = imageIterator.GetIndex();
      image->SetPixel( idx, ++val );
      ++imageIterator;
    }
  //
  ImageType< 3 >::IndexType idx = {0,0,0};
  std::cout << "image 3D (0,0,0): " << image->GetPixel( idx ) << std::endl;
  //
  //
  Writer<3>::Pointer writer = Writer<3>::New();
  writer->SetFileName( "image_3D.nii.gz" );
  writer->SetInput(image);
  //
  try
  {
    writer->Update();
  }
  catch (itk::ExceptionObject & error)
  {
    std::cerr << "error" << std::endl;
  }
  //
  //
  EXPECT_EQ( image->GetPixel( idx ), 1 );
}
//
//
// 4D image
TEST_F(ITKFunctionTest, SimpleImage4D) {
  //
  //
  using Image = itk::Image< double, 4 >;
  //
  ImageType<4>::RegionType region;
  ImageType<4>::IndexType  start;
  start = {0,0,0,0};
  //
  ImageType<4>::SizeType size;
  size = {4,4,4,4};
  //
  region.SetSize( size );
  region.SetIndex( start );
  //
  //
  ImageType<4>::Pointer image = ImageType<4>::New();
  image->SetRegions( region );
  image->Allocate();
  //
  //
  itk::ImageRegionIterator< Image > imageIterator( image, region );
  //
  double val = 0;
  while( !imageIterator.IsAtEnd() )
    {
      ImageType< 4 >::IndexType idx = imageIterator.GetIndex();
      image->SetPixel( idx, ++val );
      ++imageIterator;
    }
  //
  ImageType< 4 >::IndexType idx = {0,0,0,0};
  std::cout << "image 4D (0,0,0,0): " << image->GetPixel( idx ) << std::endl;
  //
  //
  Writer<4>::Pointer writer = Writer<4>::New();
  writer->SetFileName( "image_4D.nii.gz" );
  writer->SetInput(image);
  //
  try
  {
    writer->Update();
  }
  catch (itk::ExceptionObject & error)
  {
    std::cerr << "error" << std::endl;
  }
  //
  //
  EXPECT_EQ( image->GetPixel( idx ), 1 );
}
//
//
// 5D image
TEST_F(ITKFunctionTest, SimpleImage5D) {
  //
  //
  using Image = itk::Image< double, 5 >;
  //
  ImageType<5>::RegionType region;
  ImageType<5>::IndexType  start;
  start = {0,0,0,0,0};
  //
  ImageType<5>::SizeType size;
  size = {4,4,4,4,4};
  //
  region.SetSize( size );
  region.SetIndex( start );
  //
  //
  ImageType<5>::Pointer image = ImageType<5>::New();
  image->SetRegions( region );
  image->Allocate();
  //
  //
  itk::ImageRegionIterator< Image > imageIterator( image, region );
  //
  double val = 0;
  while( !imageIterator.IsAtEnd() )
    {
      ImageType< 5 >::IndexType idx = imageIterator.GetIndex();
      image->SetPixel( idx, ++val );
      ++imageIterator;
    }
   //
  ImageType< 5 >::IndexType idx = {0,0,0,0,0};
  std::cout << "image 5D (0,0,0,0,0): " << image->GetPixel( idx ) << std::endl;
  //
  //
  Writer<5>::Pointer writer = Writer<5>::New();
  writer->SetFileName( "image_5D.nii.gz" );
  writer->SetInput(image);
  //
  try
  {
    writer->Update();
  }
  catch (itk::ExceptionObject & error)
  {
    std::cerr << "error" << std::endl;
  }
  //
  //
  EXPECT_EQ( image->GetPixel( idx ), 1 );
}
//
//
// Read 3D image
TEST_F(ITKFunctionTest, ReadImage) {
  //
  //
  constexpr unsigned int Dimension = 3;
  //
  using ImageType = itk::Image< double, Dimension >;
  using ReaderType = itk::ImageFileReader< ImageType >;
  //
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( "image_3D.nii.gz" );
  reader->Update();
  //
  ImageType::Pointer    image  = reader->GetOutput();
  ImageType::RegionType region = image->GetLargestPossibleRegion();
  //
  //
  itk::ImageRegionIterator< ImageType > imageIterator( image, region );
  //
  while( !imageIterator.IsAtEnd() )
    {
      ImageType::IndexType idx = imageIterator.GetIndex();
      std::cout
	<< "The value: " << image->GetPixel( idx )
	<< " /or " << imageIterator.Value()
	<< " and index: " << idx
	<< std::endl;
      ++imageIterator;
    }
  //
  ImageType::IndexType idx = {0,0,0};
  std::cout << "image 3D (0,0,0): " << image->GetPixel( idx ) << std::endl;
  //
  //
  EXPECT_EQ( image->GetPixel( idx ), 1 );
}

//TEST_F(ITKFunctionTest, ByDefaultBazFalseIsFalse) {
//    LoadDataSet foo(m_bar);
//    EXPECT_EQ(foo.baz(false), false);
//}
//
//TEST_F(ITKFunctionTest, SometimesBazFalseIsTrue) {
//    LoadDataSet foo(m_bar);
//    // Have norf return true for once
//    EXPECT_CALL(m_bar,norf()).WillOnce(Return(true));
//    EXPECT_EQ(foo.baz(false), true);
//}

