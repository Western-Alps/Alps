#include "AlpsLoadDataSet.h"



//
// Allocating and initializing Singleton's static data member.
Alps::LoadDataSet* Alps::LoadDataSet::instance_ = nullptr;
//
// Constructor
Alps::LoadDataSet::LoadDataSet( const std::string JSon_file ):
  json_file_{ JSon_file }
{
  //
  // load the JSon file
  std::ifstream file_in( JSon_file.c_str() );
  file_in >> data_;

  //
  //
  //mountain_ = data_["network"]["mountain"];
  mountain_ = data_["mountain"]["name"];
}
//
//   
bool Alps::LoadDataSet::Load_ITK_images()
{
  try
    {
      
      // data_[""][""]
      //
      // 
      int number_modalities = data_["inputs"]["images"].size();
      //std::size_t image_dim = data_["network"]["Image_dim"].size();
      return true;
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << err << std::endl;
    }
}
