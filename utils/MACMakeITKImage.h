#ifndef MACMAKEITKIMAGE_H
#define MACMAKEITKIMAGE_H
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
#include "itkChangeInformationImageFilter.h"
//
//
//
#include "MACException.h"
//
//
//
namespace MAC
{
  /** \class MACMakeITKImage
   *
   * \brief 
   * D_r: number of random degres of the model.
   * D_f: number of fixed degres of the model.
   * 
   */
  class MACMakeITKImage
  {
    //
    // Some typedef
    using Image3DType = itk::Image< double, 3 >;
    using Reader3D    = itk::ImageFileReader< Image3DType >;
    using Image4DType = itk::Image< double, 4 >;
    using Reader4D    = itk::ImageFileReader< Image4DType >;
    using Writer4D    = itk::ImageFileWriter< Image4DType >;
    using MaskType    = itk::Image< unsigned char, 3 >;
        
  public:
    /** Constructor. */
  MACMakeITKImage():D_{0},image_name_{""}{};
    //
    explicit MACMakeITKImage( const long unsigned int ,
			      const std::string&,
			      const Reader3D::Pointer );
    
    /**  */
    virtual ~MACMakeITKImage(){};

    //
    // Record results
    void set_val( const std::size_t, const MaskType::IndexType, const double );
    // Write value in the image pointer
    void write();

  private:
    //
    // Dimension of the case: random or fixed
    long unsigned int D_;
    // Image name
    std::string image_name_;
    // Take the dimension of the first subject image:
    Reader3D::Pointer image_reader_;
    // Measures grouped in vector of 3D image
    std::vector< Image3DType::Pointer > images_;
  };
}
#endif
