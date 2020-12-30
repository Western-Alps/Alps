//
//
//
#include "MACException.h"
#include "AlpsMountainDummy.h"
//#include "NeuralNetwork.h"
//#include "NeuralNetworkComposite.h"
//#include "AlpsWeightsFclCPU.h"
//#include "Activations.h"
//
//
//
Alps::MountainDummy::MountainDummy()
{
};
//
//
//
void
Alps::MountainDummy::forward( std::shared_ptr< Alps::Climber > Sub )
{
  energy_ /= 2.;
};
//
//
//
void
Alps::MountainDummy::backward()
{};
