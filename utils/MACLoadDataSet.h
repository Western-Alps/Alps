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
#include "Subject.h"
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
    explicit Singleton( const std::string JSon_file );
    

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
    std::vector< Subject >& get_subjects()
      {
	return subjects_;
      }
    Subject& get_subjects( const std::string Name )
      {
      try
	{
	  int subject_found = -1;
	  for ( std::size_t s = 0 ; s < subjects_.size() ; s++ )
	    if ( subjects_[s].get_subject_name() == Name )
	      subject_found = s;

	  //
	  //
	  if ( subject_found != -1 )
	    return subjects_[ subject_found ];
	  else
	    {
	      std::string mess = "Subject " + Name + " not found.";
	      throw MAC::MACException( __FILE__, __LINE__,
				       mess.c_str(),
				       ITK_LOCATION );
	    }
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  exit(-1);
	}
      }
    const std::size_t get_number_of_input_features() const
      {
	return number_of_input_features_;
      }
    const std::size_t get_number_of_features() const
      {
	return number_of_features_;
      }
    const std::size_t get_modality_dim() const
      {
	return modality_dim_;
      }
    //
    std::string get_data_set() const
      {
	return json_file_;
      }
    //
    //
    void set_number_of_features( const int Features )
      {
	number_of_features_ = static_cast< std::size_t >( Features );
      }
    void set_data_set( const std::string JSon_file )
    {
      json_file_ = JSon_file;
    }
    //
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
    // number of features maps
    std::size_t number_of_features_{0};
    // number of input features maps
    std::size_t number_of_input_features_{0};
    // Number of images per modality
    std::size_t modality_dim_{0};
    //
    // Subjects
    std::vector< Subject > subjects_;
  };
}
#endif
