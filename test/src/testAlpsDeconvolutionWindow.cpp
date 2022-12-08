#include "testAlpsDeconvolutionWindow.h"
// ITK
#include "ITKHeaders.h"
#include "AlpsLayerTensors.h"
#include "AlpsWindow.h"

//using ::testing::Return;

DeconvolutionWindowTest::DeconvolutionWindowTest() {
    // Have qux return true by default
    //ON_CALL(m_bar,qux()).WillByDefault(Return(true));
    // Have norf return false by default
    //ON_CALL(m_bar,norf()).WillByDefault(Return(false));
}

DeconvolutionWindowTest::~DeconvolutionWindowTest() {};

void DeconvolutionWindowTest::SetUp() {};

void DeconvolutionWindowTest::TearDown() {};

TEST_F(DeconvolutionWindowTest, ByDefaultGetStatusIsTrue) {
  Alps::Window< double, 1 > window_1( /* number of kernels */{10}, 
				      /* half window size */ {1},
				      /* padding */          {1},
				      /* striding */         {1});
  //
  window_1.set_transpose( true );

  //
  //
  EXPECT_EQ( window_1.get_transpose(), true );
}
//
//
//
TEST_F(DeconvolutionWindowTest, ByDefaultGetOutputDimension) {
  //
  // Load the image
  Alps::LayerTensors< double, 2 > Subj("SimpleConvolution.nii.gz");
  // Create the kenel
  Alps::Window< double, 2 > window_1( /* number of kernels */{10}, 
				      /* half window size */ {1,1},
				      /* padding */          {0,0},
				      /* striding */         {1,1});
  //
  window_1.set_transpose( true );
  window_1.get_image_information( Subj.get_image( Alps::TensorOrder1::ACTIVATION ).get_image_region() );
  //
  //
  EXPECT_EQ( window_1.get_output_image_dimensions()[0], 6 );
}
//
//
//
TEST_F(DeconvolutionWindowTest, ByDefaultDeconvolutionPadding) {
  //
  // Load the image
  Alps::LayerTensors< double, 2 > Subj("SimpleConvolution.nii.gz");
  // Create the kenel
  Alps::Window< double, 2 > window_1( /* number of kernels */{3}, 
				      /* half window size */ {1,1},
				      /* padding */          {1,1},
				      /* striding */         {1,1});
  //
  window_1.set_transpose( true );
  window_1.get_image_information( Subj.get_image( Alps::TensorOrder1::ACTIVATION ).get_image_region() );
  //
  Eigen::SparseMatrix< int, Eigen::RowMajor > matrix = window_1.get_weights_matrix();
  std::cout << matrix << std::endl;
//  // std::cout << matrix.innerSize() << std::endl;
//  // The image load
//  Eigen::Matrix< double,  4, 1 > image_out     = Eigen::MatrixXd::Zero(  4, 1 );
//  Eigen::Matrix< double, 16, 1 > deconvolution = Eigen::MatrixXd::Zero( 16, 1 );
//  Eigen::Matrix< double, 16, 1 > image_in      = Eigen::MatrixXd::Zero( 16, 1 );
//  for ( int i = 0 ; i < 16 ; i++ )
//    {
//      std::cout << "in(" << i << ") = " << Subj[Alps::TensorOrder1::ACTIVATION][i] << std::endl;
//      image_in( i, 0 ) = Subj[Alps::TensorOrder1::ACTIVATION][i];
//    }
//  // create artificial weights
//  std::shared_ptr< double > weight_val  = std::shared_ptr< double >( new  double[10],
//								     std::default_delete< double[] >() );
//  weight_val.get()[0] = -1.; // bias whatever value
//  weight_val.get()[1] = 1.;
//  weight_val.get()[2] = 4.;
//  weight_val.get()[3] = 1.;
//  weight_val.get()[4] = 1.;
//  weight_val.get()[5] = 4.;
//  weight_val.get()[6] = 3.;
//  weight_val.get()[7] = 3.;
//  weight_val.get()[8] = 3.;
//  weight_val.get()[9] = 1.;
//  //
//  // Sparse matrix: It does not evaluate the "0". This is the reason why we can store the bias at the position
//  // [0] of the weights
//  std::cout <<" matrix.outerSize() = " << matrix.outerSize() << std::endl;
//  for (int k = 0 ; k < matrix.outerSize() ; ++k )
//    {
//      for ( typename Eigen::SparseMatrix< int, Eigen::RowMajor >::InnerIterator it( matrix, k); it; ++it)
//	{
//	  image_out( k, 0 ) += weight_val.get()[ static_cast< int >(it.value()) ] * image_in( it.index() );
//	  std::cout
//	    << "value (" << it.value() << ") in ["
//	    << it.row() << "," << it.col() << "] index: "
//	    << it.index() << " and k: " << k << std::endl; // inner index, here it is equal to it.row()
//	}
//      // add the bias
//      image_out( k, 0 ) += weight_val.get()[0];
//    }
//  //
//  std::cout << "output image: \n" << image_out << std::endl;

  
  //
  //
  //EXPECT_NEAR( image_out(0,0), 13.4286, 1.e-03 );
  EXPECT_EQ( window_1.get_output_image_dimensions()[0], 4 );
}
//
//
//
TEST_F(DeconvolutionWindowTest, ByDefaultDeconvolutionPaddingStriding) {
  //
  // Load the image
  Alps::LayerTensors< double, 2 > Subj("SimpleConvolution.nii.gz");
  // Create the kenel
  Alps::Window< double, 2 > window_1( /* number of kernels */{3}, 
				      /* half window size */ {1,1},
				      /* padding */          {1,1},
				      /* striding */         {2,2});
  //
  window_1.set_transpose( true );
  window_1.get_image_information( Subj.get_image( Alps::TensorOrder1::ACTIVATION ).get_image_region() );
  //
  Eigen::SparseMatrix< int, Eigen::RowMajor > matrix = window_1.get_weights_matrix();
  std::cout << matrix << std::endl;
//  // std::cout << matrix.innerSize() << std::endl;
//  // The image load
//  Eigen::Matrix< double,  4, 1 > image_out     = Eigen::MatrixXd::Zero(  4, 1 );
//  Eigen::Matrix< double, 16, 1 > deconvolution = Eigen::MatrixXd::Zero( 16, 1 );
//  Eigen::Matrix< double, 16, 1 > image_in      = Eigen::MatrixXd::Zero( 16, 1 );
//  for ( int i = 0 ; i < 16 ; i++ )
//    {
//      std::cout << "in(" << i << ") = " << Subj[Alps::TensorOrder1::ACTIVATION][i] << std::endl;
//      image_in( i, 0 ) = Subj[Alps::TensorOrder1::ACTIVATION][i];
//    }
//  // create artificial weights
//  std::shared_ptr< double > weight_val  = std::shared_ptr< double >( new  double[10],
//								     std::default_delete< double[] >() );
//  weight_val.get()[0] = -1.; // bias whatever value
//  weight_val.get()[1] = 1.;
//  weight_val.get()[2] = 4.;
//  weight_val.get()[3] = 1.;
//  weight_val.get()[4] = 1.;
//  weight_val.get()[5] = 4.;
//  weight_val.get()[6] = 3.;
//  weight_val.get()[7] = 3.;
//  weight_val.get()[8] = 3.;
//  weight_val.get()[9] = 1.;
//  //
//  // Sparse matrix: It does not evaluate the "0". This is the reason why we can store the bias at the position
//  // [0] of the weights
//  std::cout <<" matrix.outerSize() = " << matrix.outerSize() << std::endl;
//  for (int k = 0 ; k < matrix.outerSize() ; ++k )
//    {
//      for ( typename Eigen::SparseMatrix< int, Eigen::RowMajor >::InnerIterator it( matrix, k); it; ++it)
//	{
//	  image_out( k, 0 ) += weight_val.get()[ static_cast< int >(it.value()) ] * image_in( it.index() );
//	  std::cout
//	    << "value (" << it.value() << ") in ["
//	    << it.row() << "," << it.col() << "] index: "
//	    << it.index() << " and k: " << k << std::endl; // inner index, here it is equal to it.row()
//	}
//      // add the bias
//      image_out( k, 0 ) += weight_val.get()[0];
//    }
//  //
//  std::cout << "output image: \n" << image_out << std::endl;

  
  //
  //
  //EXPECT_NEAR( image_out(0,0), 13.4286, 1.e-03 );
  EXPECT_EQ( window_1.get_output_image_dimensions()[0], 8 );
}
