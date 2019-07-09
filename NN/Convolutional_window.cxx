#include <random>
#include <Eigen/Sparse>
//
#include "Convolutional_window.h"
#include "MACLoadDataSet.h"



//
// Constructor
MAC::Convolutional_window::Convolutional_window(): Weights()
{}
//
// Constructor
MAC::Convolutional_window::Convolutional_window( const std::string Name,
						 const int* Conv_half_window,
						 const int* Striding,
						 const int* Padding,
						 const int  Num_of_features ) : Weights(Name), previouse_conv_window_{nullptr} // no previouse conv wind
{
  //
  // members
  convolution_half_window_size_ = new int[3];
  stride_                       = new int[3];
  padding_                      = new int[3];
  //
  for ( int d = 0 ; d < 3 ; d++ )
    {
      convolution_half_window_size_[d] = Conv_half_window[d];
      stride_[d]                       = Striding[d];
      padding_[d]                      = 0;
    }
  //
  number_of_features_in_  = 1;
  number_of_features_     = number_of_features_out_ = Num_of_features;

  //
  // Initialization of the weights
  //
  
  //
  // Weights
  number_of_weights_ = 
    (2*(Conv_half_window[0]) + 1)*
    (2*(Conv_half_window[1]) + 1)*
    (2*(Conv_half_window[2]) + 1);
  // Create the random weights for each kernel
  // ToDo: Do a better initialization
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution( -1.0, 1.0 );
  // initialization
  shared_weights_ = new double*[ Num_of_features ];
  shared_biases_  = new double[ Num_of_features ];
  //
  for ( int feature = 0 ; feature < Num_of_features ; feature++ )
    {
      shared_weights_[feature] = new double[ number_of_weights_ ];
      for ( int w = 0 ; w < number_of_weights_ ; w++ )
	shared_weights_[feature][w] = distribution(generator);
      //
      shared_biases_[feature] = distribution(generator);
    }
  

  //
  // Prepare the dimensions I/O
  //

  //
  // Input dimension
  // Take the first modality first image to get the dimensions
  // ToDo: at this point we take only one input modality. Several input modalities
  //       would need to generalize the kernel to 4 dimensions
  const std::vector< Image3DType::Pointer >
    curr_images = ( MAC::Singleton::instance()->get_subjects()[0] ).get_clone_modalities_images();
  //
  Image3DType::Pointer raw_subject_image_ptr = curr_images[0];
  size_in_       = raw_subject_image_ptr->GetLargestPossibleRegion().GetSize();
  origine_in_    = raw_subject_image_ptr->GetOrigin();
  spacing_in_    = raw_subject_image_ptr->GetSpacing();
  direction_in_  = raw_subject_image_ptr->GetDirection();
  //std::cout << "In construc1" << std::endl;
  //std::cout << "size_in_ " << size_in_ << std::endl;
  //std::cout << "origine_in_ " << origine_in_ << std::endl;
  //std::cout << "spacing_in_ " << spacing_in_ << std::endl;
  //std::cout << "direction_in_ " << direction_in_ << std::endl;
  // Output dimensions
  size_out_       = feature_size( size_in_ );
  origine_out_    = feature_orig( size_in_, spacing_in_, origine_in_ );
  spacing_out_    = raw_subject_image_ptr->GetSpacing();
  direction_out_  = raw_subject_image_ptr->GetDirection();
  //std::cout << "Out construc1" << std::endl;
  //std::cout << "size_out_ " << size_out_ << std::endl;
  //std::cout << "origine_out_ " << origine_out_ << std::endl;
  //std::cout << "spacing_out_ " << spacing_out_ << std::endl;
  //std::cout << "direction_out_ " << direction_out_ << std::endl;
  //
  // Prepare the weights matrices
//toRm  Image3DType::IndexType  start = { 0, 0, 0 };
//toRm  Image3DType::RegionType region;
//toRm  region.SetSize( size_in_ );
//toRm  region.SetIndex( start );
//toRm  //
//toRm  // ToDo: tempo
//toRm  // Test image
//toRm  Reader3D::Pointer out = Reader3D::New();
//toRm  out->SetFileName( "/home/cobigo/devel/CPP/Alps/data/tempo11.nii.gz" );
//toRm  out->Update();
//toRm  Image3DType::RegionType region_out;
//toRm  region_out.SetSize( size_out_ );
//toRm  region_out.SetIndex( start );
//toRm  Image3DType::Pointer image_out = out->GetOutput();
//toRm  image_out->SetRegions( region_out );
//toRm  image_out->Allocate();
//toRm  image_out->FillBuffer( 0.0 );
//toRm  long int
//toRm    X_o = 0,
//toRm    Y_o = 0,
//toRm    Z_o = 0;
//toRm  // ToDo: tempo
//toRm  //Reader3D::Pointer reader = Reader3D::New();
//toRm  //reader->SetFileName( raw_subject_image_ptr/*->GetOutput()->GetFileName()*/ );
//toRm  //reader->Update();
  //
  // Loop over the image
  int
    half_wind_X = convolution_half_window_size_[0],
    half_wind_Y = convolution_half_window_size_[1],
    half_wind_Z = convolution_half_window_size_[2],
    //
    Im_size_X = size_in_[0],
    Im_size_Y = size_in_[1],
    Im_size_Z = size_in_[2],
    //
    stride_X = stride_[0],
    stride_Y = stride_[1],
    stride_Z = stride_[2];
  //
  std::size_t
    check_output_X = 0,
    check_output_Y = 0,
    check_output_Z = 0;
  //
  im_size_in_  = size_in_[0]*size_in_[1]*size_in_[2];
  im_size_out_ = size_out_[0]*size_out_[1]*size_out_[2];
  //
  weights_poisition_oi_ = new std::size_t*[ im_size_out_ ];
  weights_poisition_io_ = new std::size_t*[ im_size_in_ ];
  //
  // Init the I/O array: weights[in_idx][k] = out_idx
  for ( std::size_t i = 0 ; i < im_size_in_ ; i++ )
    {
      weights_poisition_io_[i] = new std::size_t[number_of_weights_];
      for ( int k = 0 ; k < number_of_weights_ ; k++ )
	// adding an unreachable position in case the position is empty
	// Do not chose zero or negative values
	weights_poisition_io_[i][k] = 999999999;
    }
  //
  // weights[out_idx][k] = in_idx
  // Creation of a sparse matrix of weight multiplication between in and out images
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  std::size_t estimation_of_entries = number_of_weights_ * im_size_out_;
  tripletList.reserve(estimation_of_entries);
  Eigen::SparseMatrix< std::size_t > W_out_in_( im_size_out_, im_size_in_ );
  W_out_in_.setZero();
  //
  std::size_t output_idx = 0;
  for ( auto Z = half_wind_Z ; Z < Im_size_Z - half_wind_Z ; Z = Z + stride_Z )
    {
      // double check output dimension
      check_output_Z++; check_output_Y = 0;
//toRm      Y_o = 0; 
      for ( auto Y = half_wind_Y ; Y < Im_size_Y - half_wind_Y ; Y = Y + stride_Y )
	{
	  // double check output dimension
	  check_output_Y++; check_output_X = 0;
//toRm	  X_o = 0; 
	  for ( auto X = half_wind_X ; X < Im_size_X - half_wind_X ; X = X + stride_X )
	    {
	      // double check output dimension
	      check_output_X++;
	      // run through the convolution window
	      int index = 0;
	      weights_poisition_oi_[ output_idx ] = new std::size_t[ number_of_weights_ ];
	      // ToDo: tempo
//toRm	      double conv = 0.;
//toRm	      Image3DType::IndexType iidx = {X_o++, Y_o, Z_o};
		      
	      //
	      for ( int z = -half_wind_Z ; z < half_wind_Z + 1 ; z++ )
		for ( int y = -half_wind_Y ; y < half_wind_Y + 1 ; y++ )
		  for ( int x = -half_wind_X ; x < half_wind_X + 1 ; x++ )
		    {
		      std::size_t in_idx = (X + x) + (Y + y)*Im_size_X + (Z + z)*Im_size_X*Im_size_Y;
		      weights_poisition_oi_[output_idx][index] = in_idx;

//toRm		      Image3DType::IndexType idx = {(X + x), (Y + y), (Z + z)};
//toRm		      std::cout
//toRm			<< "idx " << idx 
//toRm			<< " ~~ in_idx" << in_idx
//toRm			<< " ~~ raw_subject_image_ptr->GetPixel(idx) " << raw_subject_image_ptr->GetPixel(idx) 
//toRm			<< std::endl;
		      // we keep the same index to not have zero value in the sparse matrix
//toRm		      // ToDo: tempo
//toRm		      Image3DType::IndexType idx = {(X + x), (Y + y), (Z + z)};
//toRm		      conv += shared_weights_[0][index++]*raw_subject_image_ptr->GetPixel(idx);
		      tripletList.push_back(T( output_idx, in_idx, ++index ));
		      //std::cout
		      //<< "index: " << index-1 << " (" << X + x << ", " << Y + y 
		      //<< ", " << Z + z << ") -- (" << output_idx
		      //<< ", " << in_idx << ")" << std::endl;
		      //
		    }
	      //
	      output_idx++;
//toRm	      // ToDo: tempo
//toRm	      image_out->SetPixel(iidx, conv);
	    }
//toRm	  Y_o++;
	}
//toRm      Z_o++;
    }
  //
//  for ( int o = 0 ; o < im_size_out_  ; o++ )
//    for ( int k = 0 ; k < number_of_weights_; k++ )
//    std::cout
//      << "weights_poisition_oi_[" << o << "]["<< k << "] = " << weights_poisition_oi_[o][k]
//      << std::endl;
  // Fill the sparse matrix
  W_out_in_.setFromTriplets( tripletList.begin(), tripletList.end() );
  //
  for ( int k = 0 ; k < W_out_in_.outerSize() ; ++k )
    for ( Eigen::SparseMatrix< std::size_t >::InnerIterator it( W_out_in_, k ) ; it ; ++it )
      {
	  //std::cout
	  //<< " it.value() " << it.value()
	  //<< " it.row(): " << it.row()   // row index
	  //<< " it.col(): " << it.col()   // col index 
	  //<< std::endl;
	weights_poisition_io_[it.col()][it.value()-1] = it.row();
      }
  //
//  for ( int i = 0 ; i < im_size_in_  ; i++ )
//    for ( int k = 0 ; k < number_of_weights_ ; k++ )
//      std::cout
//	<< "weights_poisition_io_[" << i << "]["<< k << "] = " << weights_poisition_io_[i][k]
//	<< std::endl;


//toRm  //  //
//toRm  // ToDo: remove
//toRm  // Test matrix multiplication
//toRm  std::cout << "First step" << std::endl;
//toRm  image_to_conv = new double[ im_size_in_ ];
//toRm  image_conv    = new double[ im_size_out_ ];
//toRm  double* tempo = new double[im_size_out_];
//toRm  for ( int t = 0 ; t < im_size_out_  ; t++ )
//toRm    {
//toRm      tempo[t] = 0.;
//toRm      image_conv[t] = 0.;
//toRm    }
//toRm  for ( auto Z = 0 ; Z < Im_size_Z ; Z++ )
//toRm    {
//toRm      for ( auto Y = 0 ; Y < Im_size_Y ; Y++ )
//toRm	for ( auto X = 0 ; X < Im_size_X ; X++ )
//toRm	  {
//toRm	    Image3DType::IndexType idx = {X, Y, Z};
//toRm	    int ii = X + Y*Im_size_X + Z*Im_size_X*Im_size_Y;
//toRm	    image_to_conv[ii] =  raw_subject_image_ptr->GetPixel(idx);
//toRm//	    for ( int oo = 0 ; oo < im_size_out_ ; oo++ )
//toRm//	      {
//toRm//		for ( int k = 0 ; k < number_of_weights_ ; k++ )
//toRm//		  {
//toRm//		    if ( weights_poisition_oi_[oo][k] == ii )
//toRm//		      {
//toRm//			tempo[oo] += shared_weights_[0][k]*raw_subject_image_ptr->GetPixel(idx);
//toRm////			std::cout
//toRm////			  << "weights_poisition_oi_[" <<oo<< "][" <<k<< "] = " << weights_poisition_oi_[oo][k]
//toRm////			  << " ~~ " << ii;
//toRm////			std::cout
//toRm////			  << "shared_weights_[0][k] = " << shared_weights_[0][k]
//toRm////			  << " raw_subject_image_ptr->GetPixel(idx) " << raw_subject_image_ptr->GetPixel(idx)
//toRm////			  << " tempo[oo] = " << tempo[oo]
//toRm////			  << std::endl;
//toRm//		      }
//toRm//	  }
//toRm//    }
//toRm	  }
//toRm    }
      

  //
  //
  if ( check_output_X != size_out_[0] ||
       check_output_Y != size_out_[1] ||
       check_output_Z != size_out_[2]  )
    {
      std::string mess = "There is a dimension issue with the output: ";
      mess += "( " + std::to_string( check_output_X ) ;
      mess += ", " + std::to_string( check_output_Y ) ;
      mess += ", " + std::to_string( check_output_Z ) + ") != ";
      mess += "( " + std::to_string( size_out_[0] ) ;
      mess += ", " + std::to_string( size_out_[1] ) ;
      mess += ", " + std::to_string( size_out_[2] ) + ").";
      //
      throw MAC::MACException( __FILE__, __LINE__,
			       mess.c_str(),
			       ITK_LOCATION );
    }

//toRm  //
//toRm  // ToDo: tempo
//toRm  // Test images
//toRm  //itk::NiftiImageIO::Pointer nifti_io = itk::NiftiImageIO::New();
//toRm  //nifti_io->SetPixelType( "float" );
//toRm  //
//toRm  itk::ImageFileWriter< Image3DType >::Pointer writer = itk::ImageFileWriter< Image3DType >::New();
//toRm  writer->SetFileName( "image_test.nii.gz" );
//toRm  writer->SetInput( image_out );
//toRm  //writer->SetImageIO( nifti_io );
//toRm  writer->Update();
}
//
//
// Constructor
MAC::Convolutional_window::Convolutional_window( const std::string Name,
						 std::shared_ptr< Convolutional_window > Conv_wind,
						 const int* Conv_half_window,
						 const int* Striding,
						 const int* Padding,
						 const int  Num_of_features ) : Weights(Name), previouse_conv_window_{Conv_wind}
{
  //
  // members
  convolution_half_window_size_ = new int[3];
  stride_                       = new int[3];
  padding_                      = new int[3];
  //
  for ( int d = 0 ; d < 3 ; d++ )
    {
      convolution_half_window_size_[d] = Conv_half_window[d];
      stride_[d]                       = Striding[d];
      padding_[d]                      = 0;
    }
  //
  number_of_features_in_  = Conv_wind->get_number_of_features_out();
  number_of_features_     = number_of_features_out_ = Num_of_features;

  //
  // Initialization of the weights
  //

  //
  // WARNING: the bias represents index 0!
  number_of_weights_ = 
    (2*(Conv_half_window[0]) + 1)*
    (2*(Conv_half_window[1]) + 1)*
    (2*(Conv_half_window[2]) + 1);
  // Create the random weights for each kernel
  // ToDo: Do a better initialization
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution( -1.0, 1.0 );
  // initialization
  shared_weights_ = new double*[ Num_of_features ];
  shared_biases_  = new double[ Num_of_features ];
  //
  for ( int feature = 0 ; feature < Num_of_features ; feature++ )
    {
      shared_weights_[feature] = new double[ number_of_weights_ ];
      for ( int w = 0 ; w < number_of_weights_ ; w++ )
	shared_weights_[feature][w] = distribution(generator);
      //
      shared_biases_[feature] = distribution(generator);
    }
  

  //
  // Prepare the dimensions I/O
  //

  //
  // Input dimension
  size_in_       = Conv_wind->get_size_out();	  
  origine_in_    = Conv_wind->get_origine_out();  
  spacing_in_    = Conv_wind->get_spacing_out();  
  direction_in_  = Conv_wind->get_direction_out();
  //std::cout << "In construc2" << std::endl;
  //std::cout << "size_in_ " << size_in_ << std::endl;
  //std::cout << "origine_in_ " << origine_in_ << std::endl;
  //std::cout << "spacing_in_ " << spacing_in_ << std::endl;
  //std::cout << "direction_in_ " << direction_in_ << std::endl;
  //
  // Output dimensions
  size_out_       = feature_size( size_in_ );
  origine_out_    = feature_orig( size_in_, spacing_in_, origine_in_ );
  spacing_out_    = Conv_wind->get_spacing_out();
  direction_out_  = Conv_wind->get_direction_out();
  //std::cout << "Out construc2" << std::endl;
  //std::cout << "size_out_ " << size_out_ << std::endl;
  //std::cout << "origine_out_ " << origine_out_ << std::endl;
  //std::cout << "spacing_out_ " << spacing_out_ << std::endl;
  //std::cout << "direction_out_ " << direction_out_ << std::endl;
  //
  // Loop over the image
  int
    half_wind_X = convolution_half_window_size_[0],
    half_wind_Y = convolution_half_window_size_[1],
    half_wind_Z = convolution_half_window_size_[2],
    //
    Im_size_X = size_in_[0],
    Im_size_Y = size_in_[1],
    Im_size_Z = size_in_[2],
    //
    stride_X = stride_[0],
    stride_Y = stride_[1],
    stride_Z = stride_[2];
  //
  std::size_t
    check_output_X = 0,
    check_output_Y = 0,
    check_output_Z = 0;
  //
  im_size_in_  = size_in_[0]*size_in_[1]*size_in_[2];
  im_size_out_ = size_out_[0]*size_out_[1]*size_out_[2];
  //
  weights_poisition_oi_ = new std::size_t*[ im_size_out_ ];
  weights_poisition_io_ = new std::size_t*[ im_size_in_ ];
  //
  // Init the I/O array: weights[in_idx][k] = out_idx
  for ( std::size_t i = 0 ; i < im_size_in_ ; i++ )
    {
      weights_poisition_io_[i] = new std::size_t[number_of_weights_];
      for ( int k = 0 ; k < number_of_weights_ ; k++ )
	// adding an unreachable position in case the position is empty
	// Do not chose zero or negative values
	weights_poisition_io_[i][k] = 999999999;
    }
  //
  // weights[out_idx][k] = in_idx
  // Creation of a sparse matrix of weight multiplication between in and out images
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  std::size_t estimation_of_entries = number_of_weights_ * im_size_out_;
  tripletList.reserve(estimation_of_entries);
  Eigen::SparseMatrix< std::size_t > W_out_in_( im_size_out_, im_size_in_ );
  W_out_in_.setZero();
  //
  std::size_t output_idx = 0;
  for ( auto Z = half_wind_Z ; Z < Im_size_Z - half_wind_Z ; Z = Z + stride_Z )
    {
      // double check output dimension
      check_output_Z++; check_output_Y = 0;
      for ( auto Y = half_wind_Y ; Y < Im_size_Y - half_wind_Y ; Y = Y + stride_Y )
	{
	  // double check output dimension
	  check_output_Y++; check_output_X = 0;
	  for ( auto X = half_wind_X ; X < Im_size_X - half_wind_X ; X = X + stride_X )
	    {
	      // double check output dimension
	      check_output_X++;
	      // run through the convolution window
	      int index = 0;
	      weights_poisition_oi_[ output_idx ] = new std::size_t[ number_of_weights_ ];
	      //
	      for ( int z = -half_wind_Z ; z < half_wind_Z + 1 ; z++ )
		for ( int y = -half_wind_Y ; y < half_wind_Y + 1 ; y++ )
		  for ( int x = -half_wind_X ; x < half_wind_X + 1 ; x++ )
		    {
		      std::size_t in_idx = (X + x) + (Y + y)*Im_size_X + (Z + z)*Im_size_X*Im_size_Y;
		      weights_poisition_oi_[output_idx][index] = in_idx;
		      // we keep the same index to not have zero value in the sparse matrix
		      tripletList.push_back(T( output_idx, in_idx, ++index ));
		    }
	      //
	      output_idx++;
	    }
	}
    }
  // Fill the sparse matrix
  W_out_in_.setFromTriplets( tripletList.begin(), tripletList.end() );
  //
  for ( int k = 0 ; k < W_out_in_.outerSize() ; ++k )
    for ( Eigen::SparseMatrix< std::size_t >::InnerIterator it( W_out_in_, k ) ; it ; ++it )
	weights_poisition_io_[it.col()][it.value()-1] = it.row();
      

  //
  //
  if ( check_output_X != size_out_[0] ||
       check_output_Y != size_out_[1] ||
       check_output_Z != size_out_[2]  )
    {
      std::string mess = "There is a dimension issue with the output: ";
      mess += "( " + std::to_string( check_output_X ) ;
      mess += ", " + std::to_string( check_output_Y ) ;
      mess += ", " + std::to_string( check_output_Z ) + ") != ";
      mess += "( " + std::to_string( size_out_[0] ) ;
      mess += ", " + std::to_string( size_out_[1] ) ;
      mess += ", " + std::to_string( size_out_[2] ) + ").";
      //
      throw MAC::MACException( __FILE__, __LINE__,
			       mess.c_str(),
			       ITK_LOCATION );
    }
}
//
//
//
void
MAC::Convolutional_window::print()
{
}
//
//
//
void
MAC::Convolutional_window::check_match( Image3DType::SizeType Size_1,
					 Image3DType::SizeType Size_2 )
{
  try
    {
      if ( Size_1[0] != Size_2[0] ||
	   Size_1[1] != Size_2[1] ||
	   Size_1[2] != Size_2[2]  )
	{
	  std::string mess = "There is a dimension issue with the output: ";
	  mess += "( " + std::to_string( Size_1[0] ) ;
	  mess += ", " + std::to_string( Size_1[1] ) ;
	  mess += ", " + std::to_string( Size_1[2] ) + ") != ";
	  mess += "( " + std::to_string( Size_2[0] ) ;
	  mess += ", " + std::to_string( Size_2[1] ) ;
	  mess += ", " + std::to_string( Size_2[2] ) + ").";
	  //
	  throw MAC::MACException( __FILE__, __LINE__,
				   mess.c_str(),
				   ITK_LOCATION );
	}
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << err << std::endl;
      exit(-1);
    }
}
//
//
//
void
MAC::Convolutional_window::save_weights()
{
  //
  //
  std::ofstream out( name_, std::ios::out | std::ios::binary | std::ios::trunc );
  //
  out.write( (char*) (&number_of_features_in_),  sizeof(size_t) );
  out.write( (char*) (&number_of_features_out_), sizeof(size_t) );
  //
  for ( std::size_t feature = 0 ; feature < number_of_features_out_ ; feature++ )
    {
      out.write( (char*) shared_weights_[feature], number_of_weights_*sizeof(double) );
      out.write( (char*) &shared_biases_[feature], sizeof(double) );
    }
}
//
//
//
void
MAC::Convolutional_window::load_weights()
{
  //
  //
  std::ifstream out( name_, std::ios::in | std::ios::binary );
  //
  out.read( (char*) (&number_of_features_in_),  sizeof(size_t) );
  out.read( (char*) (&number_of_features_out_), sizeof(size_t) );
  //
  for ( std::size_t feature = 0 ; feature < number_of_features_out_ ; feature++ )
    {
      out.read( (char*) shared_weights_[feature], number_of_weights_*sizeof(double) );
      out.read( (char*) &shared_biases_[feature], sizeof(double) );
    }
}
//
//
Image3DType::SizeType
MAC::Convolutional_window::feature_size( const Image3DType::SizeType Input_size ) const
{
  //
  // Output feature size in the direction Dim
  Image3DType::SizeType to_return;
  //
  for ( int d = 0 ; d < 3 ; d++ )
    {
      to_return[d]  = ( Input_size[d] - 2 * convolution_half_window_size_[d] - 1 + 2 * padding_[d]);
      to_return[d] /= stride_[d];
      to_return[d] +=  1;
    }      

  //
  //
  return to_return;
}
//
//
Image3DType::PointType
MAC::Convolutional_window::feature_orig( const Image3DType::SizeType    Input_size,
					 const Image3DType::SpacingType Input_spacing,
					 const Image3DType::PointType   Input_orig ) const
{
  //
  // Output feature size in the direction Dim
  Image3DType::SizeType  size;
  Image3DType::PointType to_return;
  //
  for ( int d = 0 ; d < 3 ; d++ )
    {
      size[d]  = ( Input_size[d] - 2 * convolution_half_window_size_[d] - 1 + 2 * padding_[d]);
      size[d] /= stride_[d];
      size[d] +=  1;
      //
      to_return[d] = 0.5 * ( size[d] - 1 ) * Input_spacing[d];
      to_return[d] = ( Input_orig[d] > 0 ? to_return[d] : -1 * to_return[d] );
    }      

  //
  //
  return to_return;
}
//
//
MAC::Convolutional_window::~Convolutional_window()
{
  //
  // Window
  if (convolution_half_window_size_)
    delete [] convolution_half_window_size_;
  convolution_half_window_size_ = nullptr;
  if (stride_)
    delete [] stride_;
  stride_ = nullptr;
  if (padding_)
    delete [] padding_;
  padding_ = nullptr;
  // Weights position matrices
  if (weights_poisition_oi_)
    {
      for (std::size_t o = 0 ; o < im_size_out_ ; o++)
	if ( weights_poisition_oi_[o] )
	  {
	    delete [] weights_poisition_oi_[o];
	    weights_poisition_oi_[o] = nullptr;
	  }
      delete [] weights_poisition_oi_;
      weights_poisition_oi_ = nullptr;
    }
  //
  if (weights_poisition_io_)
    {
      for (std::size_t i = 0 ; i < im_size_in_ ; i++)
	if ( weights_poisition_io_[i] )
	  {
	    delete [] weights_poisition_io_[i];
	    weights_poisition_io_[i] = nullptr;
	  }
      delete [] weights_poisition_io_;
      weights_poisition_io_ = nullptr;
    }
}
