#ifndef SUBJECT_H
#define SUBJECT_H
//
//
//
#include <iostream>
#include <fstream>
#include <math.h>  
#include <map>
#include <list>
// Egen
#include <Eigen/Core>
#include <Eigen/Eigen>
//
// ITK
//
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>
#include <itkNiftiImageIO.h>
#include <itkOrientImageFilter.h>
#include <itkSpatialOrientation.h>
//
//
//
#include "Exception.h"
//
//
//
namespace MAC
{
  inline bool file_exists ( const std::string& name )
  {
    std::ifstream f( name.c_str() );
    return f.good();
  }

  /** \class Subject
   *
   * \brief 
   * Each subject loads a pointer for all modalities
   * 
   */
  class Subject
    {
      //
      // Some typedef
      using Image3DType = itk::Image< double, 3 >;
      using Reader3D    = itk::ImageFileReader< Image3DType >;
      using MaskType    = itk::Image< unsigned char, 3 >;
 
    public:
      /** Constructor. */
    Subject():
      idx_{0} {};
      //
      //explicit Subject( const int, const int );
    
      /** Destructor */
      virtual ~Subject( const int );

      //
      // Accessors
      inline const int get_idx() const { return idx_ ;}

      //
      // Write the output matrix: fitted parameters and the covariance matrix
      void write_solution(){};

      //
      // Add time point
      void add_modality(){};

    private:
      //
      // private member function
      //

      //
      // Subject parameters
      //
    
      // Identification number
      int idx_;
      // vector of modalities
      std::vector< Reader3D::Pointer > modalities_ITK_images_; 
    };

  //
  //
  //
  MAC::Subject::Subject( const int Index ):
  idx_{Index}
  {}
}
#endif
