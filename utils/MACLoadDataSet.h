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
    const std::size_t get_number_of_madalities() const
      {
	return number_of_madalities_;
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
    // number of modalities
    std::size_t number_of_madalities_{0};
    // Number of images per modality
    std::size_t modality_dim_{0};
    //
    // Subjects
    std::vector< Subject > subjects_;
  };
}
#endif
