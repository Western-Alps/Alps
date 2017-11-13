#include "MACLoadDataSet.h"



//
// Allocating and initializing Singleton's static data member.
MAC::Singleton* MAC::Singleton::instance_ = nullptr;
//
// Constructor
MAC::Singleton::Singleton( const std::string JSon_file ):
  json_file_{ JSon_file }
{
  //
  // load the JSon file
  std::ifstream file_in( JSon_file.c_str() );
  file_in >> data_;
  
  //
  // Conditions
  //

  //
  // Load images select train or test
  std::size_t number_of_labels = data_["inputs"]["labels"].size();
  if ( data_["strategy"]["status"] == "train" )
    if ( number_of_labels > 0 )
      train_ = true;
    else
      throw MAC::MACException( __FILE__, __LINE__,
			       "Training labels are missing.",
			       ITK_LOCATION );
  else
    train_ = false;

  //
  // If we have more than one set of images, all sets needs to have the same number 
  // of images.
  number_of_features_ = data_["inputs"]["images"].size();
  modality_dim_       = data_["inputs"]["images"][0].size();
  if ( data_["inputs"]["images"].size() > 1 )
    {
      for ( auto modality : data_["inputs"]["images"] )
	if ( modality_dim_ != modality.size() )
	  {
	    std::string mess = "The number of images must be the same for each modalities.\n";
	    mess += "The first modality has " + std::to_string( modality_dim_ );
	    mess += " images and another has: " + std::to_string( modality.size() );
	    throw MAC::MACException( __FILE__, __LINE__,
				     mess.c_str(),
				     ITK_LOCATION );
	  }
    }

  //
  // The number of labels should be the same as the number of images
  if ( train_ )
    if ( number_of_labels != modality_dim_ )
      {
	std::string mess = "Number of images and labels must be the same.\n";
	throw MAC::MACException( __FILE__, __LINE__,
				 mess.c_str(),
				 ITK_LOCATION );
      }

  //
  // Build vector of subjects
  //
      
  //
  // Load the subjects
  // load images
  subjects_.resize( modality_dim_ );
  for ( auto modality : data_["inputs"]["images"] )
    {
      std::cout << modality << std::endl;
      for ( int img_mod = 0 ; img_mod < static_cast< int >( modality_dim_ ) ; img_mod++ )
	{
	  subjects_[img_mod].add_modality( modality[img_mod] );
	}
    }
  // load labels
  for ( int img_mod = 0 ; img_mod < static_cast< int >( modality_dim_ ) ; img_mod++ )
    {
      subjects_[img_mod].add_label( data_["inputs"]["labels"][img_mod] );
      subjects_[img_mod].update();
      // update subject name
      subjects_[img_mod].set_subject_name( "subject_" + std::to_string( img_mod ) );
    }
}
