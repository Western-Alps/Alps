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
Alps::NeuralNetworkComposite::weight_update( std::shared_ptr< Alps::Climber > Sub )
{
  try
    {
	// 1. Reset energy cost function
	// 2. propagate
	auto rit = nn_composite_.rbegin();
	for ( ; rit != nn_composite_.rend() ; rit++ )
	  (*rit)->weight_update( Sub );
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
Alps::NeuralNetworkComposite::save_weights( std::ofstream& Weights_file ) const
{
  try
    {
      if ( Weights_file.is_open() )
	for ( auto nn_elem : nn_composite_ )
	  nn_elem->save_weights( Weights_file );
      else
	{
	  std::string mess = "The weights file has no I/O access.\n";
	  throw MAC::MACException( __FILE__, __LINE__,
				   mess.c_str(),
				   ITK_LOCATION );
	}
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
