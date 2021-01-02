#include "testImage.h"
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
  auto image_ptr = itk::ImageIOFactory::CreateImageIO( "../images/MNITS/000000-num5.png",
						       itk::CommonEnums::IOFileMode::ReadMode );
  image_ptr->SetFileName( "../images/MNITS/000000-num5.png" );
  image_ptr->ReadImageInformation();
  //
  // Read the ITK image
  typename Reader< 2 >::Pointer img_ptr = Reader< 2 >::New();
  img_ptr->SetFileName( image_ptr->GetFileName() );
  img_ptr->Update();
  //
  // Constructor of a subject
  Alps::Image< 2 > Subj = Alps::Image< 2 >( img_ptr );
  //
  //
  EXPECT_EQ( 0, 0) ;
}
// Accessors
TEST_F(ImageTest, ByDefaultImageGetZ) {
  //
  // use an image
  auto image_ptr = itk::ImageIOFactory::CreateImageIO( "../images/MNITS/000000-num5.png",
						       itk::CommonEnums::IOFileMode::ReadMode );
  image_ptr->SetFileName( "../images/MNITS/000000-num5.png" );
  image_ptr->ReadImageInformation();
  //
  // Read the ITK image
  typename Reader< 2 >::Pointer img_ptr = Reader< 2 >::New();
  img_ptr->SetFileName( image_ptr->GetFileName() );
  img_ptr->Update();
  //
  // Constructor of a subject
  Alps::Image< 2 > Subj = Alps::Image< 2 >( img_ptr );
  //
  // Get the Z array
  //  int s = Subj.get_array_size();
  //  for ( int i = 0 ; i < s ; i++ )
  //    std::cout << "ZZ["<<i<<"]" << Subj.get_z()[i] << std::endl;

  //
  //
  EXPECT_EQ( ( Subj.get_z().get() )[206], 253 );
}
// Accessors
TEST_F(ImageTest, ByDefaultImageSetZ) {
  //
  //
  std::shared_ptr< double > zz = std::shared_ptr< double >( new double[3], std::default_delete< double[] >() );
  ( zz.get() )[0] = 99.99;
  ( zz.get() )[1] = 99.99;
  ( zz.get() )[2] = 99.99;
  //
  // use an image
  auto image_ptr = itk::ImageIOFactory::CreateImageIO( "../images/MNITS/000000-num5.png",
						       itk::CommonEnums::IOFileMode::ReadMode );
  image_ptr->SetFileName( "../images/MNITS/000000-num5.png" );
  image_ptr->ReadImageInformation();
  //
  // Read the ITK image
  typename Reader< 2 >::Pointer img_ptr = Reader< 2 >::New();
  img_ptr->SetFileName( image_ptr->GetFileName() );
  img_ptr->Update();
  //
  // Constructor of a subject
  Alps::Image< 2 > Subj = Alps::Image< 2 >( img_ptr );
  //
  // Set the Z array
  Subj.set_z( zz );

  //
  //
  EXPECT_EQ( ( Subj.get_z().get() )[0], 99.99) ;
}
// Accessors
TEST_F(ImageTest, ByDefaultImageSetZ2) {
  //
  //
  std::vector< double > zz = std::vector< double >(3, 99.99);
  //
  // use an image
  auto image_ptr = itk::ImageIOFactory::CreateImageIO( "../images/MNITS/000000-num5.png",
						       itk::CommonEnums::IOFileMode::ReadMode );
  image_ptr->SetFileName( "../images/MNITS/000000-num5.png" );
  image_ptr->ReadImageInformation();
  //
  // Read the ITK image
  typename Reader< 2 >::Pointer img_ptr = Reader< 2 >::New();
  img_ptr->SetFileName( image_ptr->GetFileName() );
  img_ptr->Update();
  //
  // Constructor of a subject
  Alps::Image< 2 > Subj = Alps::Image< 2 >( img_ptr );
  //
  // Set the Z array
  Subj.set_z( zz );

  //
  //
  EXPECT_EQ( ( Subj.get_z().get() )[0], 99.99) ;
}
// Accessors
TEST_F(ImageTest, ByDefaultImageSetEps) {
  //
  //
  std::shared_ptr< double > Eps = std::shared_ptr< double >( new double[3], std::default_delete< double[] >() );
  ( Eps.get() )[0] = 99.99;
  ( Eps.get() )[1] = 99.99;
  ( Eps.get() )[2] = 99.99;
  //
  // use an image
  auto image_ptr = itk::ImageIOFactory::CreateImageIO( "../images/MNITS/000000-num5.png",
						       itk::CommonEnums::IOFileMode::ReadMode );
  image_ptr->SetFileName( "../images/MNITS/000000-num5.png" );
  image_ptr->ReadImageInformation();
  //
  // Read the ITK image
  typename Reader< 2 >::Pointer img_ptr = Reader< 2 >::New();
  img_ptr->SetFileName( image_ptr->GetFileName() );
  img_ptr->Update();
  //
  // Constructor of a subject
  Alps::Image< 2 > Subj = Alps::Image< 2 >( img_ptr );
  //
  // Set the eps array
  Subj.set_eps( Eps );

  //
  //
  EXPECT_EQ( ( Subj.get_eps().get() )[0], 99.99) ;
}
// Accessors
TEST_F(ImageTest, ByDefaultImageSetEps2) {
  //
  //
  std::vector< double > Eps = std::vector< double >(3, 99.99);
  //
  // use an image
  auto image_ptr = itk::ImageIOFactory::CreateImageIO( "../images/MNITS/000000-num5.png",
						       itk::CommonEnums::IOFileMode::ReadMode );
  image_ptr->SetFileName( "../images/MNITS/000000-num5.png" );
  image_ptr->ReadImageInformation();
  //
  // Read the ITK image
  typename Reader< 2 >::Pointer img_ptr = Reader< 2 >::New();
  img_ptr->SetFileName( image_ptr->GetFileName() );
  img_ptr->Update();
  //
  // Constructor of a subject
  Alps::Image< 2 > Subj = Alps::Image< 2 >( img_ptr );
  //
  // Set the eps array
  Subj.set_eps( Eps );

  //
  //
  EXPECT_EQ( ( Subj.get_eps().get() )[0], 99.99) ;
}
//// Add modalities
//TEST_F(ImageTest, ByDefaultSubjectAddModalitiesTrue) {
//  // Constructor of a subject
//  Alps::Image< 2 > Subj = Alps::Image< 2 >( 0, 2);
//  // load modalities
//  Subj.add_modalities("../images/MNITS/000000-num5.png");
//  Subj.add_modalities("../images/MNITS/000000-num5.png");
//  //
//  //
//  EXPECT_EQ( Subj.check_modalities(), true) ;
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

