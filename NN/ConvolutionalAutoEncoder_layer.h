#ifndef CONVOLUTIONALAUTOENCODER_LAYER_H
#define CONVOLUTIONALAUTOENCODER_LAYER_H
//
//
//
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <math.h>  
#include <map>
#include <list>
#include <memory>
#include <random>
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

//
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
#include "MACLoadDataSet.h"
#include "Subject.h"
#include "NeuralNetwork.h"
#include "Weights.h"
//
// CUDA
//
#include <cuda_runtime.h>

//
//
namespace MAC
{
  /** \class ConvolutionalAutoEncoder_layer
   *
   * \brief 
   * 
   * 
   */
  template< class ActivationFunction  >
    class ConvolutionalAutoEncoder_layer : public Convolutional_layer< ActivationFunction >
    {

    public:

      /** Constructor encoder. */
      explicit ConvolutionalAutoEncoder_layer( const std::string,
					       const int, 
					       const int,
					       const int* );
      /** Constructor decoder. */
      explicit ConvolutionalAutoEncoder_layer( const std::string,
					       const int,
					       const int*,
					       std::shared_ptr< MAC::NeuralNetwork > );
      /** Destructor */
      virtual ~ConvolutionalAutoEncoder_layer();

      //
      // Initialization
      virtual void initialization(){};
      //
      // get the layer name
      virtual std::string get_layer_name(){ return Convolutional_layer< ActivationFunction >::get_layer_name();};
      //
      // get the layer name
      virtual Layer get_layer_type(){ return layer_type_;};
      //
      // Forward propagation
      virtual void forward( Subject&, const Weights& W = Weights() );
      //
      //
      virtual void backward(){};
      //
      // Backward error propagation
      virtual void backward_error_propagation(){};
      //
      //
      virtual void add( std::shared_ptr< NeuralNetwork > ){};
      //
      //
      virtual int get_number_weights() const{return Convolutional_layer< ActivationFunction >::get_number_weights();};

    private:
      Layer layer_type_;
      };
  //
  //
  //
  // 
  //
  //
  //
  /*
    Layer_name,   
    Layer_number
    Shrinkage: if it is positive it downsample, otherwise it up sample,  
    Do_we_pool, 
    Window_size 
  */
  template< class A >
    MAC::ConvolutionalAutoEncoder_layer< A >::ConvolutionalAutoEncoder_layer( const std::string Layer_name,
									      const int         Layer_number,
									      const int         Shrinkage,
									      const int*        Window_size ):
  MAC::Convolutional_layer< A >::Convolutional_layer( Layer_name, Layer_number, Shrinkage, Window_size ),
    layer_type_{ convolutional_encoder }
  {
  };
  //
  //
  //
  /*
    Layer_name,   
    Layer_number
    Window_size 
  */
  template< class A >
    MAC::ConvolutionalAutoEncoder_layer< A >::ConvolutionalAutoEncoder_layer( const std::string Layer_name,
									      const int         Layer_number,
									      const int*        Window_size,
									      std::shared_ptr< MAC::NeuralNetwork > Encoder ):
  MAC::Convolutional_layer< A >::Convolutional_layer( Layer_name,
						      Layer_number,
						      0,
						      Window_size ),
    layer_type_{ convolutional_decoder }
  {
  };
  //
  //
  //
  template< class A > void
    MAC::ConvolutionalAutoEncoder_layer< A >::forward( Subject& Sub, const Weights& W )
    {
      try
	{
	  switch( layer_type_ )
	    {
	    case convolutional_encoder:
	      {
		Convolutional_layer< A >::forward( Sub );
		break;
	      }
	    case convolutional_decoder:
	      {
		break;
	      }
	    default:
	      throw MAC::MACException( __FILE__, __LINE__,
				       "this function can only be used for stacked convolutional auto-encoder.",
				       ITK_LOCATION );
	    }
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  exit(-1);
	}
    };
  //
  //
  //
  template< class A >
    MAC::ConvolutionalAutoEncoder_layer< A >::~ConvolutionalAutoEncoder_layer() 
    {
    }
}
#endif
