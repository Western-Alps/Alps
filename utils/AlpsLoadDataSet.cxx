#include "AlpsLoadDataSet.h"



//
// Allocating and initializing Singleton's static data member.
MAC::LoadDataSet* MAC::LoadDataSet::instance_ = nullptr;
//
// Constructor
MAC::LoadDataSet::LoadDataSet( const std::string JSon_file ):
  json_file_{ JSon_file }
{
  //
  // load the JSon file
  std::ifstream file_in( JSon_file.c_str() );
  file_in >> data_;
}
