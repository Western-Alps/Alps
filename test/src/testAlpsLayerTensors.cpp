#include "testAlpsLayerTensors.h"
#include <iostream>
#include <memory>
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
// ITK
#include "ITKHeaders.h"
#include "AlpsLayerTensors.h"

//using ::testing::Return;

LayerTensorsTest::LayerTensorsTest() {
    // Have qux return true by default
    //ON_CALL(m_bar,qux()).WillByDefault(Return(true));
    // Have norf return false by default
    //ON_CALL(m_bar,norf()).WillByDefault(Return(false));
}

LayerTensorsTest::~LayerTensorsTest() {};

void LayerTensorsTest::SetUp() {};

void LayerTensorsTest::TearDown() {};
//
// Constructor
TEST_F(LayerTensorsTest, ByDefaultLayerTensorsZero) {
  //
  // Constructor of a subject
  Alps::LayerTensors< double, 2 > Subj("../test/images/MNITS/000000-num5.png");
  //
  //
  EXPECT_EQ( 0, 0) ;
}
// operator[]
TEST_F(LayerTensorsTest, ByDefaultLayerTensorsOperator) {
  //
  //
  Alps::LayerTensors< double, 2 > Subj("../test/images/MNITS/000000-num5.png");
  //
  //
	 EXPECT_EQ( Subj[Alps::TensorOrder1::ACTIVATION][206], 253 );
}
// operator[]
TEST_F(LayerTensorsTest, ByDefaultLayerTensorsImage) {
  //
  //
  Alps::LayerTensors< double, 2 > Subj("../test/images/MNITS/000000-num5.png");
  //
  std::cout << "Region: \n" << Subj.get_image( Alps::TensorOrder1::ACTIVATION ).get_image_region() << std::endl;
  std::cout << "Start: \n" << Subj.get_image( Alps::TensorOrder1::ACTIVATION ).get_image_start() << std::endl;
  std::cout << "Size: \n" << Subj.get_image( Alps::TensorOrder1::ACTIVATION ).get_image_size() << std::endl;
  
  //
  //
	 EXPECT_EQ( Subj[Alps::TensorOrder1::ACTIVATION][206], 253 );
}
// operator() -- hadamard
TEST_F(LayerTensorsTest, ByDefaultLayerTensorsHadamard) {
  //
  //
  std::vector< std::size_t > size(1);
  size[0] = 5;
  //
  std::vector< double > activation  = std::vector< double >( 5, 0. );
  std::vector< double > derivative  = std::vector< double >( 5, 0. );
  std::vector< double > error       = std::vector< double >( 5, 0. );
  std::vector< double > werror      = std::vector< double >( 5, 0. );
  //
  for ( int i = 0 ; i < 5 ; i++ )
    {
      activation[i] = static_cast<double>( i );
      derivative[i] = static_cast<double>( i * 10 );
      error[i]      = static_cast<double>( i * 0.1 );
      werror[i]     = static_cast<double>( i * 0.01 );
    }
  //
  std::array< std::vector< double >, 4 > current = { activation, derivative, error, werror };
    
  Alps::LayerTensors< double, 2 > Subj( size, current );
  //
  // Test the hadamart product
  std::vector< double > hadamart = std::move( Subj(Alps::TensorOrder1::ACTIVATION,
						   Alps::TensorOrder1::DERIVATIVE) );
  //
  for ( int i = 0 ; i < 5 ; i++ )
    std::cout << "hadamart("<<i<<") = " << hadamart[i] << std::endl;
  
  //
  //
  EXPECT_EQ( hadamart[4], 160 );
}
//// Accessors
//TEST_F(LayerTensorsTest, ByDefaultImageSet) {
//  //
//  //
//  std::vector< double > zz = std::vector< double >( new double[3], std::default_delete< double[] >() );
//  ( zz.get() )[0] = 99.99;
//  ( zz.get() )[1] = 99.99;
//  ( zz.get() )[2] = 99.99;
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

//TEST_F(LayerTensorsTest, ByDefaultBazFalseIsFalse) {
//    Image foo(m_bar);
//    EXPECT_EQ(foo.baz(false), false);
//}
//
//TEST_F(LayerTensorsTest, SometimesBazFalseIsTrue) {
//    Image foo(m_bar);
//    // Have norf return true for once
//    EXPECT_CALL(m_bar,norf()).WillOnce(Return(true));
//    EXPECT_EQ(foo.baz(false), true);
//}

