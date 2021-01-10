#ifndef ALPSLOADDATASET_H
#define ALPSLOADDATASET_H
//
//
//
#include <iostream> 
#include <fstream>
#include <string>
//
// JSON interface
//
#include "nlohmann/json.hpp"
using json = nlohmann::json;
//
// 
//
#include "MACException.h"
//
//
//
/*! \namespace Alps
 *
 * Name space Alps.
 *
 */
namespace Alps
{
  /*! \class LoadDataSet
   *
   * \brief class loading all the data set.    
   *
   */
  class LoadDataSet
  {
    //
    // Constructor
    explicit LoadDataSet( const std::string JSon_file );
    

  public:
    //
    // Accessors
    const bool          get_status() const
    {
      return true;
    }
    const json&         get_data() const
    {
      return data_;
    }
    const std::string   get_mountain() const
    {
      return mountain_;
    }
    //
    std::string         get_data_set() const
      {
	return json_file_;
      }
    //
    //
    void                set_data_set( const std::string JSon_file )
    {
      json_file_ = JSon_file;
    }


    //
    // Load the ITK images
    bool                Load_ITK_images();

    //
    //
    void                print_data_set() const
    {
      std::cout << data_.dump(4) << std::endl;
    }
    
    //
    // Single instance
    static LoadDataSet* instance( const std::string JSon_file = "" )
    {
      if ( !instance_ )
	instance_ = new LoadDataSet( JSon_file );
      //
      return instance_;
    }


  private:
    //! Unique instance
    static LoadDataSet *instance_;
    //! path to the jason file
    std::string         json_file_;
    //! json data
    json                data_;

    //! What type of algorithm:
    std::string         mountain_{""};
  };
}
#endif
