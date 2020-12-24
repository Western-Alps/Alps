#include "AlpsWeightsFclCPU.h"
//
//
Alps::WeightsFclCPU::WeightsFclCPU(  std::shared_ptr< Alps::Mountain > Layer,
				     const std::vector< int >          Layer_size,
				     const std::vector< int >          Prev_layer_size  ):
  layer_{Layer}
{
  try
    {
      //
      // Random create the variables between [-1,1]
//      weights_            = std::make_shared<Eigen::MatrixXd>( Eigen::MatrixXd::Random(Layer_size[0],
//										       Prev_layer_size[0] + 1 /* biais */) );
//      weights_transposed_ = std::make_shared<Eigen::MatrixXd>( weights_->transpose() );
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << err << std::endl;
    }
};
  
