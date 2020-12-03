#ifndef ALPSSUBJECTCPU_H
#define ALPSSUBJECTCPU_H
//
//
//
#include <iostream> 
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
//
// 
//
#include "MACException.h"
#include "AlpsSubject.h"
#include "AlpsTools.h"
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
  /*! \class SubjectCPU
   *
   * \brief class SubjectCPU record the information 
   * of the subject through the processing on CPU 
   * architecture.
   *
   */
  template< class Function, int Dim >
  class SubjectCPU : Alps::Subject
  {
    // Some typedef
    using ImageType = itk::Image< double, Dim >;
    using Reader    = itk::ImageFileReader< ImageType >;
    using MaskType  = itk::Image< unsigned char, Dim >;

  public:
    //
    /** Constructor */
    explicit SubjectCPU( const int, const int );
    /* Destructor */
    virtual ~SubjectCPU(){};

    //
    // Accessors


    //
    // functions
    void add_modalities();

  private:
    //
    // Subject information
    int subject_number_{0};
    // number of modalities
    int number_modalities_{0};

    //
    // Vector of modalities 
    std::vector< ImageType::Pointer > modalities_;
    // This function is the continuous step function
    Function activation_function_;
  };
  
  //
  // Constructor
  template< class F, int Dim >
  Alps::SubjectCPU<F,Dim>::SubjectCPU( const int SubNumber,
				        const int NumModalities ):
    subject_number_{SubNumber}, number_modalities_{NumModalities}
  {
  }
  //
  // 
  template< class F, int Dim > void
  Alps::SubjectCPU<F,Dim>::add_modalities()
  {
    try
      {
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
      }
  }
}
#endif
