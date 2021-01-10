#include "AlpsWeightsFclCPU.h"
//
//
//
Alps::WeightsFclCPU::WeightsFclCPU(  std::shared_ptr< Alps::Mountain > Layer,
				     const std::vector< std::size_t >  Layer_size,
				     const std::vector< std::size_t >  Prev_layer_size  ):
  layer_{Layer}
{
  try
    {
      //
      // Get the number of layers defined
      std::size_t
	prev_num_of_layers    = Prev_layer_size.size(),
	current_num_of_layers = Layer_size.size();
      //
      //
      int
	current_nodes  = 0,
	previous_nodes = 0;
      // How many nodes we had in this layer:
      for ( std::size_t l = 0 ; l < current_num_of_layers ; l++ )
	current_nodes += Layer_size[l];
      // How many nodes we had in the previous layer:
      for ( std::size_t pl = 0 ; pl < prev_num_of_layers ; pl++ )
	previous_nodes += Prev_layer_size[pl];
      // Random create the variables between [-1,1]
      weights_ = std::make_shared< Eigen::MatrixXd >(Eigen::MatrixXd::Random( current_nodes,
									      previous_nodes + 1 /* biais */) );
      std::cout
	<< "Weights ["<<weights_->rows()<<"x"<<weights_->cols()<<"]" 
	<< std::endl;
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << err << std::endl;
    }
};
//
//
//
std::shared_ptr< double >
Alps::WeightsFclCPU::activate( std::vector< Alps::LayerTensors< double, 2 > >& Image_tensors,
			       std::shared_ptr< Alps::Function >               Activation_object )
{
  //
  // Check the dimensions are right
  long int tensors_size = 0;
  for ( auto tensor : Image_tensors )
    tensors_size += static_cast< long int >( tensor.get_tensor_size()[0] );
  //
  if ( weights_->cols() != tensors_size + 1 )
    {
      std::string
	mess = std::string("There is miss match between the number of columns (")
	+ std::to_string( weights_->cols() )
	+ std::string(") and the size of the input tensor (")
	+ std::to_string( tensors_size )
	+ std::string("+1).");
      throw MAC::MACException( __FILE__, __LINE__,
			       mess.c_str(),
			       ITK_LOCATION );
    }
  //
  // Converter the tensor into an Eigen matrix
  std::shared_ptr< double > z_out = std::shared_ptr< double >( new  double[weights_->rows()],
							       std::default_delete<  double[] >() );
  Eigen::MatrixXd           a_out = Eigen::MatrixXd::Zero( weights_->rows(), 1 );
  Eigen::MatrixXd           z_in  = Eigen::MatrixXd::Zero( weights_->cols(), 1 );
  // Load the tensor image into a Eigen vector
  std::size_t shift = 1;
  z_in(0,0) = 1.; // bias
  for ( auto tensor : Image_tensors )
    {
      std::size_t img_size = tensor.get_tensor_size()[0];
      for ( std::size_t s = 0 ; s < img_size ; s++ )
	z_in(s+shift,0) = tensor[TensorOrder1::NEURONS][s];
      shift += img_size;
    }
  // process
  a_out = *( weights_.get() ) * z_in;
  // Apply the activation function
  long int activation_size = weights_->rows();
  for ( long int s = 0 ; s < activation_size ; s++ )
    z_out.get()[s] = Activation_object->f( a_out(s,0) );

  //
  //
  return z_out;
};
  
