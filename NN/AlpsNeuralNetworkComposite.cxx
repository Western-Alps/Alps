//
//
//
#include "MACException.h"
#include "AlpsNeuralNetworkComposite.h"
//
//
//
void
Alps::NeuralNetworkComposite::forward( std::shared_ptr< Alps::Climber > Sub )
{
  try
    {
      //
      for ( auto nn_elem : nn_composite_ )
	nn_elem->forward( Sub );
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << err << std::endl;
      exit(EXIT_FAILURE);
    }
}
//
//
//
void
Alps::NeuralNetworkComposite::backward( std::shared_ptr< Alps::Climber > Sub )
{
  try
    {
	// 1. Reset energy cost function
	// 2. propagate
	auto rit = nn_composite_.rbegin();
	for ( ; rit != nn_composite_.rend() ; rit++ )
	  (*rit)->backward( Sub );
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << err << std::endl;
      exit(EXIT_FAILURE);
    }
}
//
//
//
void
Alps::NeuralNetworkComposite::add( std::shared_ptr< Alps::Layer > NN )
{
  try
    {
      nn_composite_.push_back( NN );
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << err << std::endl;
      exit(EXIT_FAILURE);
    }
}
