#include "testAlpsConvolutionWindow.h"
// ITK
#include "ITKHeaders.h"
#include "AlpsLayerTensors.h"
#include "AlpsWindow.h"

//using ::testing::Return;

ConvolutionWindowTest::ConvolutionWindowTest() {
    // Have qux return true by default
    //ON_CALL(m_bar,qux()).WillByDefault(Return(true));
    // Have norf return false by default
    //ON_CALL(m_bar,norf()).WillByDefault(Return(false));
}

ConvolutionWindowTest::~ConvolutionWindowTest() {};

void ConvolutionWindowTest::SetUp() {};

void ConvolutionWindowTest::TearDown() {};

TEST_F(ConvolutionWindowTest, ByDefaultGetStatusIsTrue) {
  Alps::Window< double, 1 > window_1( /* number of kernels */{10}, 
				      /* half window size */ {1},
				      /* padding */          {1},
				      /* striding */         {1});
  EXPECT_EQ(true,true);
}


TEST_F(ConvolutionWindowTest, ByDefaultGetOutputDimension) {
  //
  // Load the image
  Alps::LayerTensors< double, 2 > Subj("SimpleConvolution.nii.gz");
  // Create the kenel
  Alps::Window< double, 2 > window_1( /* number of kernels */{10}, 
				      /* half window size */ {1,1},
				      /* padding */          {0,0},
				      /* striding */         {1,1});
  //
  window_1.get_image_information( Subj.get_image( Alps::TensorOrder1::ACTIVATION ).get_image_region() );
  //
  //
  //std::cout << "Output image size: \n" << window_1.get_output_image_dimensions()[0] << std::endl;
  
  //
  //
  EXPECT_EQ( window_1.get_output_image_dimensions()[0], 4 );
}

//TEST_F(ConvolutionWindowTest, ByDefaultBazFalseIsFalse) {
//    LoadDataSet foo(m_bar);
//    EXPECT_EQ(foo.baz(false), false);
//}
//
//TEST_F(ConvolutionWindowTest, SometimesBazFalseIsTrue) {
//    LoadDataSet foo(m_bar);
//    // Have norf return true for once
//    EXPECT_CALL(m_bar,norf()).WillOnce(Return(true));
//    EXPECT_EQ(foo.baz(false), true);
//}

