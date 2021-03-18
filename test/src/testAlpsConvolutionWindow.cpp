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
//
//
//
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
  EXPECT_EQ( window_1.get_output_image_dimensions()[0], 2 );
}
//
//
//
TEST_F(ConvolutionWindowTest, ByDefaultConvolutionDeconvolution) {
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
  Eigen::SparseMatrix< double, Eigen::RowMajor > matrix = window_1.get_weights_matrix();
  std::cout << matrix << std::endl;
  std::cout << matrix.innerSize() << std::endl;
  // The image load
  Eigen::Matrix< double,  4, 1 > image_out     = Eigen::MatrixXd::Zero(  4, 1 );
  Eigen::Matrix< double, 16, 1 > deconvolution = Eigen::MatrixXd::Zero( 16, 1 );
  Eigen::Matrix< double, 16, 1 > image_in      = Eigen::MatrixXd::Zero( 16, 1 );
  for ( int i = 0 ; i < 16 ; i++ )
    {
      std::cout << "in(" << i << ") = " << Subj[Alps::TensorOrder1::ACTIVATION][i] << std::endl;
      image_in( i, 0 ) = Subj[Alps::TensorOrder1::ACTIVATION][i];
    }
  // create artificial weights
  std::shared_ptr< double > weight_val  = std::shared_ptr< double >( new  double[10],
								     std::default_delete< double[] >() );
  weight_val.get()[0] = 99.; // bias whatever value
  weight_val.get()[1] = 1.;
  weight_val.get()[2] = 4.;
  weight_val.get()[3] = 1.;
  weight_val.get()[4] = 1.;
  weight_val.get()[5] = 4.;
  weight_val.get()[6] = 3.;
  weight_val.get()[7] = 3.;
  weight_val.get()[8] = 3.;
  weight_val.get()[9] = 1.;
  //
  for (int k = 0 ; k < matrix.outerSize() ; ++k )
    for ( typename Eigen::SparseMatrix< double, Eigen::RowMajor >::InnerIterator it( matrix, k); it; ++it)
      {
	image_out( k, 0 ) += weight_val.get()[ static_cast< int >(it.value()) ] * image_in( it.index() );
	std::cout
	  << "value (" << it.value() << ") in ["
	  << it.row() << "," << it.col() << "] inedx: "
	  << it.index() << std::endl; // inner index, here it is equal to it.row()
      }
  //
  std::cout << "output image: \n" << image_out << std::endl;

  //
  // Deconvolution or transpose convolution
  Eigen::SparseMatrix< double, Eigen::RowMajor > transposed = matrix.transpose();
  std::cout << transposed << std::endl;
  std::cout << transposed.outerSize() << std::endl;
  //
  for (int k = 0 ; k < transposed.outerSize() ; ++k )
    for ( typename Eigen::SparseMatrix< double, Eigen::RowMajor >::InnerIterator it( transposed, k ); it; ++it)
      {
	deconvolution( k, 0 ) += weight_val.get()[ static_cast< int >(it.value()) ] * image_in( it.index() );
	std::cout
	  << "value (" << it.value() << ") in ["
	  << it.row() << "," << it.col() << "] inedx: "
	  << it.index() << std::endl; // inner index, here it is equal to it.row()
      }
  //
  std::cout << "Deconvolution (transposed conv) image: \n" << deconvolution << std::endl;

 
  //
  // Deconvolution or transpose convolution on a random input
  Eigen::Matrix< double,  4, 1 > image_rnd = Eigen::MatrixXd::Zero(  4, 1 );
  image_rnd(0,0) = 2.;
  image_rnd(1,0) = 1.;
  image_rnd(2,0) = 4.;
  image_rnd(3,0) = 4.;
  //
  deconvolution = Eigen::MatrixXd::Zero( 16, 1 );
  for (int k = 0 ; k < transposed.outerSize() ; ++k )
    for ( typename Eigen::SparseMatrix< double, Eigen::RowMajor >::InnerIterator it( transposed, k ); it; ++it)
      {
	deconvolution( k, 0 ) += weight_val.get()[ static_cast< int >(it.value()) ] * image_rnd( it.index() );
	std::cout
	  << "value (" << it.value() << ") in ["
	  << it.row() << "," << it.col() << "] inedx: "
	  << it.index()
	  << " KxImg = " << weight_val.get()[ static_cast< int >(it.value()) ]
	  << " x " << image_rnd( it.index() )
	  << std::endl; // inner index, here it is equal to it.row()
      }
  //
  std::cout << "Deconvolution (transposed conv) image: \n" << deconvolution << std::endl;
 

  //
  //
  //std::cout << "Output image size: \n" << window_1.get_output_image_dimensions()[0] << std::endl;
  
  //
  //
  EXPECT_EQ( deconvolution(11,0), 13. );
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

