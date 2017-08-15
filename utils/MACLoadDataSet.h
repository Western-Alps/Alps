#ifndef SINGLETON_H
#define SINGLETON_H
//
//
//
#include <iostream> 
#include <fstream>
#include <string>
//
// JSON interface
//
#include "json.hpp"
using json = nlohmann::json;
//
// 
//
#include "MACException.h"
//
//
//
/*! \namespace Classification
 *
 * Name space for classification package.
 *
 */
namespace MAC
{
  /*! \class Classifier
   * \brief class representing    
   *
   */
  class Singleton
  {
    //
    // Constructor
    explicit Singleton( const std::string JSon_file ):json_file_{ JSon_file }
    {
      //
      // load the JSon file
      std::ifstream file_in( JSon_file.c_str() );
      file_in >> data_;

      //
      // Conditions
      //

      //
      // If we don't have label images: it is a use (train_ = 0); otherwise: train (train_ = 1)
      int number_of_labels = 0;
      if ( !data_["inputs"]["labels"].size() )
	train_ = false;
      else
	number_of_labels = data_["inputs"]["labels"].size();

      //
      // If we have more than one set of images, all sets needs to have the same number 
      // of images.
      int modality_dim = data_["inputs"]["images"][0].size();
      if ( data_["inputs"]["images"].size() > 1 )
	{
	  for ( auto modality : data_["inputs"]["images"] )
	    if ( modality_dim != modality.size() )
	    {
	      std::string mess = "The number of images must be the same for each modalities.\n";
	      mess += "The first modality has " + std::to_string( modality_dim );
	      mess += " images and another has: " + std::to_string( modality.size() );
	      throw MAC::MACException( __FILE__, __LINE__,
				       mess.c_str(),
				       ITK_LOCATION );
	    }
	}

      //
      // The number of labels should be the same as the number of images
      if ( train_ )
	if ( number_of_labels != modality_dim )
	  {
	    std::string mess = "Number of images and labels must be the same.\n";
	    throw MAC::MACException( __FILE__, __LINE__,
				     mess.c_str(),
				     ITK_LOCATION );
	  }
    }
    

  public:
    //
    // Accessors
     bool get_status() const
      {
	return train_;
      }
    const json& get_data() const
      {
	return data_;
      }
    //
    std::string get_data_set() const
      {
	return json_file_;
      }
    //
    void set_data_set( const std::string JSon_file )
    {
      json_file_ = JSon_file;
    }
    //
    void print_data_set()
    {
      // explicit conversion to string
      //std::string s = data_.dump();
      // serialization with pretty printing
      // pass in the amount of spaces to indent
      std::cout << data_.dump(4) << std::endl;
    }
    
    //
    // Single instance
    static Singleton* instance( const std::string JSon_file = "" )
    {
      if ( !instance_ )
	instance_ = new Singleton( JSon_file );
      return instance_;
    }


  private:
    //! Unique instance
    static Singleton *instance_;
    //! path to the jason file
    std::string json_file_;
    // json data
    json data_;
    // Status for trainning or using the algorithms
    // if the label is empty, the status should be automaticly "false".
    bool train_{true};
  };

//
// Allocating and initializing Singleton's static data member.
Singleton* Singleton::instance_ = nullptr;
}
#endif
