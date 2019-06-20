#ifndef DECONVOLUTIONAL_WINDOW_H
#define DECONVOLUTIONAL_WINDOW_H
//
//
//
#include <iostream> 
#include <fstream>
#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <random>//
// ITK
//
#include <itkSize.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkConstNeighborhoodIterator.h>
#include <itkImageRegionIterator.h>
#include <itkNiftiImageIO.h>
#include <itkOrientImageFilter.h>
#include <itkSpatialOrientation.h>
#include "itkChangeInformationImageFilter.h"
#include "itkImageDuplicator.h"
// {down,up}sampling the pooling image and upsampling
#include "itkIdentityTransform.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkShrinkImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkReLUInterpolateImageFunction.h"
// Some typedef
using Image3DType    = itk::Image< double, 3 >;
using Reader3D       = itk::ImageFileReader< Image3DType >;
using Writer3D       = itk::ImageFileWriter< Image3DType >;
using MaskType       = itk::Image< unsigned char, 3 >;
using FilterType     = itk::ChangeInformationImageFilter< Image3DType >;
using DuplicatorType = itk::ImageDuplicator< Image3DType > ;
using ShrinkImageFilterType = itk::ShrinkImageFilter < Image3DType, Image3DType >;

//
// 
//
#include "MACException.h"
#include "Weights.h"
#include "Convolutional_window.h"
#include "MACLoadDataSet.h"
//
//
//
/*! \namespace MAC
 *
 * Name space for MAC.
 *
 */
namespace MAC
{
  /*! \class Deconvolutional_window
   * \brief class representing the shared weights in a convolutional layer.
   *
   * convolution_half_window_size_: 
   * 	The window size is a 3 dimensions, the dimension must be odd! because we are
   * 	taking in account the center of the convolution.
   * 	3 dimensions for x,y and z; 
   * 	Note: To complete the kernel size, we need to know how many feature maps we 
   *    had in the previouse round.
   *
   * stride_:
   * 	The stride array represents the jump of a kernel center for the next convolution.
   *
   * padding_:
   * 	ToDo: This feature is ignored for the first version of the software.
   * 
   * transpose_:
   *    This Window can be the deconvolution process of a convolution window or 
   *    simply an up sampling.
   *
   *  number_of_features_:  
   * 	Number of features: 2,4,8,16,...,1024. Represents the number of kernel we
   *    are going to build.
   *
   *
   *
   */
  class Deconvolutional_window  : public Weights  
  {
  public:
    //
    // Constructor
    Deconvolutional_window();
    //
    // Constructor
    Deconvolutional_window( const std::string,
			    std::shared_ptr< Convolutional_window > );
    //
    // Destructor
    virtual ~Deconvolutional_window();
    
    
  public:
    //
    // Accessors
    virtual void print();
    // Save the weightd
    virtual void save_weights(){};
    // Save the weightd
    virtual void load_weights(){};


  private:
    // 
    // Inputs
    std::shared_ptr< Convolutional_window > previouse_conv_window_;
    // Half window
    int*                                    convolution_half_window_size_;
    int*                                    stride_;
    int*                                    padding_;

    //
    // Outputs
    // Number of features: 2,4,8,16,...,1024
    std::size_t number_of_features_in_;
    std::size_t number_of_features_out_;
    //
    // Image information input layer
    Image3DType::SizeType      size_in_;
    Image3DType::SpacingType   spacing_in_;
    Image3DType::PointType     origine_in_;
    Image3DType::DirectionType direction_in_;
    // Image information output layer
    Image3DType::SizeType      size_out_;
    Image3DType::SpacingType   spacing_out_;
    Image3DType::PointType     origine_out_;
    Image3DType::DirectionType direction_out_;
  };
}
#endif