//
//
//
#include "MACException.h"
#include "AlpsNeuralNetworkComposite.h"

//
//
//
Alps::NeuralNetworkComposite::NeuralNetworkComposite()
{
  try
    {}
  catch( itk::ExceptionObject & err )
    {
      std::cerr << err << std::endl;
      exit(EXIT_FAILURE);
    }
}
//
//
//
Alps::NeuralNetworkComposite::~NeuralNetworkComposite()
{
  try
    {}
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
Alps::NeuralNetworkComposite::forward( std::shared_ptr< Alps::Climber > Sub )
{
  try
    {
      //
      for ( auto nn_elem : nn_composite_ )
	{
	  std::cout << "New elem" << std::endl;
	  nn_elem->forward( Sub );
	  // we just want to save the last energy
	  // energies are already cumulated in the container
	  // energy_ = nn_elem->get_energy();
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
Alps::NeuralNetworkComposite::backward()
{
  try
    {
//	// 1. Reset energy cost function
//	// 2. propagate
//	std::list< std::shared_ptr< NeuralNetwork > >::reverse_iterator rit = nn_composite_.rbegin();
//	for ( ; rit != nn_composite_.rend() ; rit++ )
//	  {
//	    std::cout << "Bkw elem" << std::endl;
//	    (*rit)->backward();
//	    (*rit)->backward_error_propagation();
//	  }
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
