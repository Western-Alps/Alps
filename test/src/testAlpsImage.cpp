#include "testAlpsImage.h"
#include <iostream>
#include <memory>
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
// ITK
#include "ITKHeaders.h"
#include "AlpsImage.h"

//using ::testing::Return;

ImageTest::ImageTest() {
    // Have qux return true by default
    //ON_CALL(m_bar,qux()).WillByDefault(Return(true));
    // Have norf return false by default
    //ON_CALL(m_bar,norf()).WillByDefault(Return(false));
}

ImageTest::~ImageTest() {};

void ImageTest::SetUp() {};

void ImageTest::TearDown() {};
//
// Constructor
TEST_F(ImageTest, ByDefaultImageZero) {
  //
  // use an image
  auto image_ptr = itk::ImageIOFactory::CreateImageIO( "../test/images/MNITS/000000-num5.png",
						       itk::CommonEnums::IOFileMode::ReadMode );
  image_ptr->SetFileName( "../test/images/MNITS/000000-num5.png" );
  image_ptr->ReadImageInformation();
  //
  // Read the ITK image
  typename Reader< 2 >::Pointer img_ptr = Reader< 2 >::New();
  img_ptr->SetFileName( image_ptr->GetFileName() );
  img_ptr->Update();
  //
  // Constructor of a subject
  Alps::Image< double, 2 > Subj = Alps::Image< double, 2 >( img_ptr );
  //
  //
  EXPECT_EQ( 0, 0) ;
}
// Accessors
TEST_F(ImageTest, ByDefaultImageGetZ) {
  //
  // use an image
  auto image_ptr = itk::ImageIOFactory::CreateImageIO( "../test/images/MNITS/000000-num5.png",
						       itk::CommonEnums::IOFileMode::ReadMode );
  image_ptr->SetFileName( "../test/images/MNITS/000000-num5.png" );
  image_ptr->ReadImageInformation();
  //
  // Read the ITK image
  typename Reader< 2 >::Pointer img_ptr = Reader< 2 >::New();
  img_ptr->SetFileName( image_ptr->GetFileName() );
  img_ptr->Update();
  //
  // Constructor of a subject
  Alps::Image< double, 2 > Subj = Alps::Image< double, 2 >( img_ptr );

  std::cout << "Region: \n" << Subj.get_image_region() << std::endl;
  std::cout << "Start: \n" << Subj.get_image_start() << std::endl;
  std::cout << "Size: \n" << Subj.get_image_size() << std::endl;
  //
  // Get the Z array
  //  int s = Subj.get_array_size();
  //  for ( int i = 0 ; i < s ; i++ )
  //    std::cout << "ZZ["<<i<<"]" << Subj.get_tensor()[i] << std::endl;

  //
  //
  EXPECT_EQ( Subj.get_tensor()[206], 253 );
}
//// Accessors
//TEST_F(ImageTest, ByDefaultImageSetZ) {
//  //
//  //
//  std::vector< double > zz = std::vector< double >( 3, 0. );
//  zz[0] = 99.99;
//  zz[1] = 99.99;
//  zz[2] = 99.99;
//  //
//  // use an image
//  auto image_ptr = itk::ImageIOFactory::CreateImageIO( "../images/MNITS/000000-num5.png",
//						       itk::CommonEnums::IOFileMode::ReadMode );
//  image_ptr->SetFileName( "../images/MNITS/000000-num5.png" );
//  image_ptr->ReadImageInformation();
//  //
//  // Read the ITK image
//  typename Reader< 2 >::Pointer img_ptr = Reader< 2 >::New();
//  img_ptr->SetFileName( image_ptr->GetFileName() );
//  img_ptr->Update();
//  //
//  // Constructor of a subject
//  Alps::Image< double, 2 > Subj = Alps::Image< double, 2 >( img_ptr );
//  //
//  // Set the Z array
//  Subj.set_tensor( zz );
//
//  //
//  //
//  EXPECT_EQ( ( Subj.get_tensor().get() )[0], 99.99) ;
//}

//TEST_F(ImageTest, ByDefaultBazFalseIsFalse) {
//    Image foo(m_bar);
//    EXPECT_EQ(foo.baz(false), false);
//}
//
//TEST_F(ImageTest, SometimesBazFalseIsTrue) {
//    Image foo(m_bar);
//    // Have norf return true for once
//    EXPECT_CALL(m_bar,norf()).WillOnce(Return(true));
//    EXPECT_EQ(foo.baz(false), true);
//}

