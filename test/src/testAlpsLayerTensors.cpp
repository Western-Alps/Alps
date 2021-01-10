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
  Alps::LayerTensors< double, 2 > Subj("../images/MNITS/000000-num5.png");
  //
  //
  EXPECT_EQ( 0, 0) ;
}
// Accessors
TEST_F(LayerTensorsTest, ByDefaultLayerTensorsOperator) {
  //
  //
  Alps::LayerTensors< double, 2 > Subj("../images/MNITS/000000-num5.png");
  //
  //
	 EXPECT_EQ( Subj[Alps::TensorOrder1::NEURONS][206], 253 );
}
//// Accessors
//TEST_F(LayerTensorsTest, ByDefaultImageSet) {
//  //
//  //
//  std::shared_ptr< double > zz = std::shared_ptr< double >( new double[3], std::default_delete< double[] >() );
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

