#include "AlpsWeightsFclCPU.h"
//
//
Alps::WeightsFclCPU::WeightsFclCPU( std::shared_ptr< Alps::Mountain > Layer,
				    const std::vector< int >          Layer_size,
				    const std::vector< int >          Prev_layer_size ):
  layer_{Layer}
{
  //
  // Random create the variables between [-1,1]
  weights_            = Eigen::MatrixXd::Random( Layer_size[0],
						 Prev_layer_size[0] );
  weights_transposed_ = weights_.transpose();
};
