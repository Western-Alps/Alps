#include "AlpsWeightsFclCPU.h"
//
//
//
Alps::WeightsFclCPU::WeightsFclCPU(  std::shared_ptr< Alps::Mountain > Layer,
				     const std::vector< int >          Layer_size,
				     const std::vector< int >          Prev_layer_size  ):
  layer_{Layer}
{
  try
    {
//      //
//      // Get the number of layers defined
//      std::size_t
//	prev_num_of_layers    = Prev_layer_size.size(),
//	current_num_of_layers = Layer_size.size();
//      //
//      // For each current layer, we create a set of weights linking to the previous layers
//      for ( std::size_t l = 0 ; l < current_num_of_layers ; l++ )
//	{
//	  //
//	  // How many nodes we had in the previous layer:
//	  int nodes = 0;
//	  for ( std::size_t pl = 0 ; pl < prev_num_of_layers ; pl++ )
//	    nodes += Prev_layer_size[pl];
//	  // Random create the variables between [-1,1]
//	  Eigen::MatrixXd weights = Eigen::MatrixXd::Random( Layer_size[l], nodes + 1 /* biais */ );
//	  std::cout
//	    << "Weights ["<<weights.rows()<<"x"<<weights.cols()<<"]" 
//	    << std::endl;
//	  // record
//	  weights_.push_back( weights );
//	}
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << err << std::endl;
    }
};
//
//
//
void
Alps::WeightsFclCPU::activate( std::vector< std::shared_ptr< Alps::Image< double, 1 > > >& Image )
{
  try
    {
//      std::cout
//	<< "" << Tensors[0]->get_array_size()
//	<< std::endl;
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << err << std::endl;
    }
};
  
