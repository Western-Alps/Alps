#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <vector>
#include <chrono>
#include <memory>
//
// JSON interface
//
#include "nlohmann/json.hpp"
using json = nlohmann::json;
//
// ITK
//
#include <itkImageFileReader.h>
#include <itkSpatialOrientationAdapter.h>
#include "itkChangeInformationImageFilter.h"
//
// 
//
#include "MACException.h"
#include "MACLoadDataSet.h"
//#include "CrossValidation_k_folds.h"
#include "Mont_Maudit_builder.h"
//
//using Validation = MAC::CrossValidation_k_folds< MAC::Mont_Maudit_builder >;
//
//
//
class InputParser{
public:
  explicit InputParser ( const int &argc, const char **argv )
  {
    for( int i = 1; i < argc; ++i )
      tokens.push_back( std::string(argv[i]) );
  }
  //
  const std::string getCmdOption( const std::string& option ) const
  {
    //
    //
    std::vector< std::string >::const_iterator itr = std::find( tokens.begin(),
								tokens.end(),
								option );
    if ( itr != tokens.end() && ++itr != tokens.end() )
      return *itr;

    //
    //
    return "";
  }
  //
  bool cmdOptionExists( const std::string& option ) const
  {
    return std::find( tokens.begin(), tokens.end(), option) != tokens.end();
  }
private:
  std::vector < std::string > tokens;
};
//
//
//
int
main( const int argc, const char **argv )
{
  try
    {
      //
      // Parse the arguments
      //
      if( argc > 1 )
	{
	  InputParser input( argc, argv );
	  if( input.cmdOptionExists("-h") )
	    throw MAC::MACException( __FILE__, __LINE__,
				     "./Alps -c data_set.json",
				     ITK_LOCATION );

	  //
	  // takes the json file
	  const std::string& filename = input.getCmdOption("-c");
	  //
	  if ( !filename.empty() )
	    {
	      //
	      // Load the data set
	      MAC::Singleton::instance( filename );
	      // print the data set
	      MAC::Singleton::instance()->print_data_set();

	      ////////////////////////////
	      ///////              ///////
	      ///////  PROCESSING  ///////
	      ///////              ///////
	      ////////////////////////////


	      //
	      // Task progress: elapse time
	      using ms           = std::chrono::milliseconds;
	      using get_time     = std::chrono::steady_clock ;
	      auto  start_timing = get_time::now();

	      ///////////
	      // Start //
	      ///////////

	      // -> Validation cross_val;
	      
	      //
	      // Main object
	      MAC::Mont_Maudit_builder network;
	      //// Forward
	      network.forward( MAC::Singleton::instance()->get_subjects()[0] );
	      network.forward( MAC::Singleton::instance()->get_subjects()[1] );
	      //network.forward( MAC::Singleton::instance()->get_subjects()[2] );
	      //network.forward( MAC::Singleton::instance()->get_subjects()[3] );
	      ////
	      std::cout << "Epoque energy: " << network.get_energy() << std::endl;
	      //// Backward
	      network.backward();

	      /////////
	      // End //
	      /////////

	      //
	      // Task progress
	      // End the elaps time
	      auto end_timing  = get_time::now();
	      auto diff        = end_timing - start_timing;
	      std::cout << "Process Elapsed time is :  " << std::chrono::duration_cast< ms >(diff).count()
			<< " ms "<< std::endl;

	    }
	  else
	    throw MAC::MACException( __FILE__, __LINE__,
				     "./WMH_classification -c data_set.json",
				     ITK_LOCATION );
	}
      else
	throw MAC::MACException( __FILE__, __LINE__,
				 "./WMH_classification -c data_set.json",
				 ITK_LOCATION );
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << err << std::endl;
      return EXIT_FAILURE;
    }

  //
  //
  //
  return EXIT_SUCCESS;
}
