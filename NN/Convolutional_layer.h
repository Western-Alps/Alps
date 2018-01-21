#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H
//
//
//
#include <iostream>
#include <fstream>
#include <math.h>  
#include <map>
#include <list>
#include <memory>
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
// Downsampling the pooling image and upsampling
#include "itkIdentityTransform.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkShrinkImageFilter.h"
#include "itkResampleImageFilter.h"
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

  /** \class Convolutional_layer
   *
   * \brief 
   * 
   * 
   */
  class Convolutional_layer : public NeuralNetwork
    {

    public:

      /** Constructor. */
      Convolutional_layer( const std::string,
			   const int, 
			   const int,
			   const bool,
			   const int* );
      //
      //explicit Subject( const int, const int );

      /** Destructor */
      virtual ~Convolutional_layer();

      //
      // Initialization
      virtual void initialization(){};
      //
      // get the layer name
      virtual std::string get_layer_name(){ return layer_name_;};
      //
      // get the layer name
      virtual Layer get_layer_type(){ return convolutional_layer;};
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
      virtual int get_number_weights() const { return number_of_weights_; };
      //
      //
      const std::vector< Image3DType::Pointer > pooling();
      //
      //
      std::vector< Image3DType::SizeType > pooling_image_size();
      //
      //
      const std::vector< Image3DType::Pointer > resample();
      //
      // 
      void write() const
      {
	//
	// Check
	int mod = 0;
	for (auto img_ptr : convolution_images_ )
	  {
	    itk::NiftiImageIO::Pointer nifti_io = itk::NiftiImageIO::New();
	    //
	    itk::ImageFileWriter< Image3DType >::Pointer writer =
	      itk::ImageFileWriter< Image3DType >::New();
	    //
	    std::string name = "convolutional_" + layer_name_ + "_" + std::to_string(mod) + ".nii.gz";
	    writer->SetFileName( name );
	    writer->SetInput( img_ptr );
	    writer->SetImageIO( nifti_io );
	    writer->Update();
	    //
	    mod++;
	  }
      };

    private:
      //
      // Convolutional layer's name
      std::string layer_name_;
      
      //
      // Weights
      const int  layer_number_;
      // Shrinkage for the pooling layer
      int        shrink_;
      bool       pooling_operation_{true};
      int        number_of_weights_{1};
      double*    weights_{nullptr};
      
      //
      // Size of the convolution window
      int* convolution_window_size_{new int[4]};
      int* convolution_half_window_size_;


      //
      //
      int num_of_previous_features_{0};
      // Measures grouped in vector of 3D image
      // Convolution image: vector for each modality
      std::vector< Image3DType::Pointer > convolution_images_;
      // Pulling image: vector for each modality
      std::vector< Image3DType::Pointer > pull_images_;

      //
      // Neurons, activations and delta
      std::map< std::string, std::tuple<
	std::vector< std::shared_ptr<double> > /* activations */,
	std::vector< std::shared_ptr<double> > /* neurons */,
	std::vector< std::shared_ptr<double> > /* deltas */ > > neurons_;
    };
}
#endif
