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
TEST_F(ConvolutionWindowTest, ByDefaultConvolution) {
  //
  // Load the image
  Alps::LayerTensors< double, 2 > Subj("SimpleConvolution.nii.gz");
  // Create the kenel
  Alps::Window< double, 2 > window_1( /* number of kernels */{3}, 
				      /* half window size */ {1,1},
				      /* padding */          {0,0},
				      /* striding */         {1,1});
  //
  window_1.get_image_information( Subj.get_image( Alps::TensorOrder1::ACTIVATION ).get_image_region() );
  //
  Eigen::SparseMatrix< int, Eigen::RowMajor > matrix = window_1.get_weights_matrix();
  std::cout << matrix << std::endl;
  // std::cout << matrix.innerSize() << std::endl;
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
  weight_val.get()[0] = -1.; // bias whatever value
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
  // Sparse matrix: It does not evaluate the "0". This is the reason why we can store the bias at the position
  // [0] of the weights
  std::cout <<" matrix.outerSize() = " << matrix.outerSize() << std::endl;
  for (int k = 0 ; k < matrix.outerSize() ; ++k )
    {
      for ( typename Eigen::SparseMatrix< int, Eigen::RowMajor >::InnerIterator it( matrix, k); it; ++it)
	{
	  image_out( k, 0 ) += weight_val.get()[ static_cast< int >(it.value()) ] * image_in( it.index() );
	  std::cout
	    << "value (" << it.value() << ") in ["
	    << it.row() << "," << it.col() << "] index: "
	    << it.index() << " and k: " << k << std::endl; // inner index, here it is equal to it.row()
	}
      // add the bias
      image_out( k, 0 ) += weight_val.get()[0];
    }
  //
  std::cout << "output image: \n" << image_out << std::endl;

  
  //
  //
  EXPECT_NEAR( image_out(0,0), 13.4286, 1.e-03 );
}
//
//
//
TEST_F(ConvolutionWindowTest, ByDefaultConvolutionPadding) {
  //
  // Load the image
  Alps::LayerTensors< double, 2 > Subj("SimpleConvolution.nii.gz");
  // Create the kenel
  Alps::Window< double, 2 > window_1( /* number of kernels */{3}, 
				      /* half window size */ {1,1},
				      /* padding */          {1,1},
				      /* striding */         {1,1});
  //
  window_1.get_image_information( Subj.get_image( Alps::TensorOrder1::ACTIVATION ).get_image_region() );
  //
  Eigen::SparseMatrix< int, Eigen::RowMajor > matrix = window_1.get_weights_matrix();
  std::cout << matrix << std::endl;
  
  //
  //
  EXPECT_EQ( window_1.get_output_image_dimensions()[0], 4 );
}
//
//
//
TEST_F(ConvolutionWindowTest, ByDefaultConvolutionPaddingStriding) {
  //
  // Load the image
  Alps::LayerTensors< double, 2 > Subj("SimpleConvolution.nii.gz");
  // Create the kenel
  Alps::Window< double, 2 > window_1( /* number of kernels */{3}, 
				      /* half window size */ {1,1},
				      /* padding */          {1,1},
				      /* striding */         {2,2});
  //
  window_1.get_image_information( Subj.get_image( Alps::TensorOrder1::ACTIVATION ).get_image_region() );
  //
  Eigen::SparseMatrix< int, Eigen::RowMajor > matrix = window_1.get_weights_matrix();
  std::cout << matrix << std::endl;

  //
  //
  EXPECT_EQ( window_1.get_output_image_dimensions()[0], 2 );
}
//
//
//
TEST_F(ConvolutionWindowTest, ByDefaultConvolutionWeightSave) {
  //
  // Load the image
  Alps::LayerTensors< double, 2 > Subj("SimpleConvolution.nii.gz");
  // Create the kenel
  Alps::Window< double, 2 > window_1( /* number of kernels */{3}, 
				      /* half window size */ {1,1},
				      /* padding */          {1,1},
				      /* striding */         {2,2});
  //
  window_1.get_image_information( Subj.get_image( Alps::TensorOrder1::ACTIVATION ).get_image_region() );
  //
  Eigen::SparseMatrix< int, Eigen::RowMajor > matrix_in = window_1.get_weights_matrix();
  Eigen::SparseMatrix< int, Eigen::RowMajor > matrix_out;
  std::cout << "Sparse Matrix in" << std::endl;
  std::cout << matrix_in << std::endl;

  //
  //
  std::ofstream out( "matrices_weights.txt", std::ofstream::out | std::ios::binary | std::ios::trunc );
  // Cover the layers
  window_1.save_weights( out );
  out.close();

  //
  //
  std::ifstream Weights_file("matrices_weights.txt");
  int k, number_weights;
  std::vector< std::vector< double > > weight_values;
  //
  Weights_file.read( (char*) (&k), sizeof(int) );
  Weights_file.read( (char*) (&number_weights), sizeof(int) );
  std::cout
    << "k " << k
    << "\n number_weights " << number_weights
    << std::endl;
  //
  // read the  weights
  weight_values.resize(k);
  for ( int kk = 0 ; kk < k ; kk++ )
    {
      weight_values[kk].resize(number_weights);
      Weights_file.read( (char*)&weight_values[kk][0], number_weights * sizeof(double) );
      //
      for ( int w = 0 ; w < number_weights ; w++ )
	std::cout << "weight_values["<<kk<<"]["<<w<<"] " << weight_values[kk][w] << std::endl;
    }
  //
  // Read the position
  typename Eigen::SparseMatrix< int, Eigen::RowMajor >::Index rows, cols, nnz, inSz, outSz;
  typename Eigen::SparseMatrix< int, Eigen::RowMajor >::Index sizeScalar = static_cast< typename Eigen::SparseMatrix< int, Eigen::RowMajor >::Index>( sizeof(typename Eigen::SparseMatrix< int, Eigen::RowMajor >::Scalar) );
  typename Eigen::SparseMatrix< int, Eigen::RowMajor >::Index sizeIndex  = static_cast< typename Eigen::SparseMatrix< int, Eigen::RowMajor >::Index>( sizeof(typename Eigen::SparseMatrix< int, Eigen::RowMajor >::Index) );
  typename Eigen::SparseMatrix< int, Eigen::RowMajor >::Index sizeIndexS = static_cast< typename Eigen::SparseMatrix< int, Eigen::RowMajor >::Index>( sizeof(typename Eigen::SparseMatrix< int, Eigen::RowMajor >::StorageIndex) );
  std::cout << sizeScalar << " " << sizeIndex << std::endl;
  Weights_file.read(reinterpret_cast<char*>(&rows ), sizeIndex);
  Weights_file.read(reinterpret_cast<char*>(&cols ), sizeIndex);
  Weights_file.read(reinterpret_cast<char*>(&nnz  ), sizeIndex);
  Weights_file.read(reinterpret_cast<char*>(&outSz), sizeIndex);
  Weights_file.read(reinterpret_cast<char*>(&inSz ), sizeIndex);
  //
  matrix_out.resize(rows, cols);
  matrix_out.makeCompressed();
  matrix_out.resizeNonZeros(nnz);
  //
  Weights_file.read(reinterpret_cast<char*>(matrix_out.valuePtr())     , sizeScalar * nnz  );
  Weights_file.read(reinterpret_cast<char*>(matrix_out.outerIndexPtr()), sizeIndexS * outSz);
  Weights_file.read(reinterpret_cast<char*>(matrix_out.innerIndexPtr()), sizeIndexS * nnz );
  //
  matrix_out.finalize();
  //
  //
  std::cout << "Sparse Matrix out" << std::endl;
  std::cout << matrix_out << std::endl;
  std::cout << "matrix_in(3,15) "   << matrix_in.coeff(3,15)<< std::endl;
  std::cout << "matrix_out(3,15) "  << matrix_out.coeff(3,15) << std::endl;

  //
  //
  EXPECT_EQ( matrix_in.coeff(3,15), matrix_out.coeff(3,15) );
}
