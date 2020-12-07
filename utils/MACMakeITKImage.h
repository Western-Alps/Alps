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
// ITK
#include "ITKHeaders.h"
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
  public:
    /** Constructor. */
  MACMakeITKImage():D_{0},image_name_{""}{};
    //
    explicit MACMakeITKImage( const long unsigned int ,
			      const std::string&,
			      const Reader<3>::Pointer );
    
    /**  */
    virtual ~MACMakeITKImage(){};

    //
    // Record results
    void set_val( const std::size_t, const MaskType<3>::IndexType, const double );
    // Write value in the image pointer
    void write();

  private:
    //
    // Dimension of the case: random or fixed
    long unsigned int D_;
    // Image name
    std::string image_name_;
    // Take the dimension of the first subject image:
    Reader<3>::Pointer image_reader_;
    // Measures grouped in vector of 3D image
    std::vector< ImageType<3>::Pointer > images_;
  };
}
#endif
