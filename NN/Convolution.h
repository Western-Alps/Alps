#ifndef CONVOLUTION_H
#define CONVOLUTION_H
//
//
//
#include <iostream>
#include <fstream>
#include <math.h>  
#include <map>
#include <list>
#include <memory>
#include <random>
//
// CUDA
//
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
//#include "Convolution_CUDA.cuh"
//
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

//
// Some typedef
using Image3DType    = itk::Image< double, 3 >;
using Reader3D       = itk::ImageFileReader< Image3DType >;
using Writer3D       = itk::ImageFileWriter< Image3DType >;
using MaskType       = itk::Image< unsigned char, 3 >;
using FilterType     = itk::ChangeInformationImageFilter< Image3DType >;
using DuplicatorType = itk::ImageDuplicator< Image3DType > ;
using ShrinkImageFilterType = itk::ShrinkImageFilter < Image3DType, Image3DType >;
using ConvolutionWindowType = itk::Size< 3 >;
//
//
//
#include "MACException.h"
#include "MACLoadDataSet.h"
#include "Subject.h"
#include "NeuralNetwork.h"
#include "Weights.h"
#include "Convolutional_window.h"
#include "Deconvolutional_window.h"
//
// CUDA
//
#include <cuda_runtime.h>

//
//
namespace MAC
{

  /** \class Convolution
   *
   * \brief 
   * 
   * 
   */
  template< class Grad, class ActivationFunction  >
    class Convolution : public NeuralNetwork
    {

    public:

      /** Constructor. */
      explicit Convolution( const std::string,
			    const int, 
			    std::shared_ptr< Convolutional_window > );
      /** Constructor. */
      explicit Convolution( const std::string,
			    const int, 
			    std::shared_ptr< Deconvolutional_window > );
      /** Destructor */
      virtual ~Convolution();

      //
      // Initialization
      virtual void initialization(){};
      //
      // get the layer name
      virtual std::string get_layer_name(){ return layer_name_;};
      // get the layer type
      virtual Layer get_layer_type(){ return convolution;};
      // get the energy
      virtual double get_energy(){ return energy_;};
      // Forward propagation
      virtual void forward( Subject&, const Weights& W = Weights() );
      // Backward propagation
      virtual void backward(){};
      // Backward error propagation
      virtual void backward_error_propagation(){};
      //
      // For the builders
      virtual void add( std::shared_ptr< NeuralNetwork > ){};
      // ToDo: do we need that?
      virtual int get_number_weights() const { return 0 /*number_of_weights_*/; };
      //
      //
      //Convolution_CUDA& get_cuda(){ return cuda_treatment_; };
      //
      // 
      void write() const
      {
	//
	// Check
	int mod = 0;
	for (auto img_ptr : Convolution_images_ )
	  {
	    itk::NiftiImageIO::Pointer nifti_io = itk::NiftiImageIO::New();
	    //
	    itk::ImageFileWriter< Image3DType >::Pointer writer =
	      itk::ImageFileWriter< Image3DType >::New();
	    //
	    std::string name = "Convolutional_" + layer_name_ + "_" + std::to_string(mod) + ".nii.gz";
	    writer->SetFileName( name );
	    writer->SetInput( img_ptr );
	    writer->SetImageIO( nifti_io );
	    writer->Update();
	    //
	    mod++;
	  }
      };

    protected:
      //
      // Inputs
      //
      
      // 
      // Virtual inputs
      // Convolutional layer's name
      std::string layer_name_;
      // layer energy
      double energy_{0.};
      // Weights
      const int  layer_number_;

      //
      // Convolution window
      std::shared_ptr< Convolutional_window >   window_;
      // Convolution window
      std::shared_ptr< Deconvolutional_window > dec_window_;

      //
      //
      int num_of_previous_features_{0};
      // Measures grouped in vector of 3D image
      // Convolution image: vector for each modality
      std::vector< Image3DType::Pointer > Convolution_images_;

      //
      // Neurons, activations and delta
      std::map< std::string, std::tuple<
	std::vector< std::shared_ptr<double> > /* activations */,
	std::vector< std::shared_ptr<double> > /* neurons */,
	std::vector< std::shared_ptr<double> > /* deltas */ > > neurons_;

      //
      // Image information at the lu level
      Image3DType::SizeType      size_lu_;
      Image3DType::SpacingType   spacing_lu_;
      Image3DType::PointType     origine_lu_;
      Image3DType::DirectionType direction_lu_;

      //
      // Activation function
      ActivationFunction activation_;
      // Gradient method
      Grad               gradient_;

      //
      // Cuda treatment
      //Convolution_CUDA cuda_treatment_;
    };
  //
  //
  //
  // 
  //
  //
  //
  template< class G, class A >
    MAC::Convolution< G, A >::Convolution( const std::string   Layer_name,
					   const int           Layer_number,
					   std::shared_ptr< Convolutional_window > Window ):
  MAC::NeuralNetwork::NeuralNetwork(),
    layer_name_{Layer_name}, layer_number_{Layer_number}, window_{Window}
			       
  {
  };
  //
  //
  //
  template< class G, class A >
    MAC::Convolution< G, A >::Convolution( const std::string Layer_name,
					   const int         Layer_number,
					std::shared_ptr< Deconvolutional_window > Window ):
  MAC::NeuralNetwork::NeuralNetwork(),
    layer_name_{Layer_name}, layer_number_{Layer_number}, dec_window_{Window}
			       
  {
  };
  //
  //
  //
  template< class G, class A > void
    MAC::Convolution< G, A >::forward( Subject& Sub, const Weights& W )
    {
    };
  //
  //
  //
  template< class G, class A >
    MAC::Convolution< G, A >::~Convolution() 
    {
    }
}
#endif
