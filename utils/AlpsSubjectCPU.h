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
#include "ITKHeaders.h"
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
  template< /*class Function,*/ int Dim >
  class SubjectCPU : public Alps::Subject
  {
  public:
    //
    /** Constructor */
    explicit SubjectCPU( const int, const std::size_t );
    /* Destructor */
    virtual ~SubjectCPU(){};

    //
    // Accessors
    // Subject information
    int get_subject_number()    const {return subject_number_;};
    // number of modalities
    int get_number_modalities() const {return number_modalities_;};

    //
    // functions
    virtual void add_modalities( const std::string ) override;
    virtual bool check_modalities() const override { return (number_modalities_ == modalities_.size() ? true : false);};

  private:
    //
    // Subject information
    int subject_number_{0};
    // number of modalities
    std::size_t number_modalities_{0};

    //
    // Vector of modalities 
    std::vector< typename ImageType< Dim >::Pointer >  modalities_;
    // images size
    std::vector< typename ImageType< Dim >::SizeType > modality_size_;
    // This function is the continuous step function
    /*Function                                         activation_function_;*/
  };
  
  //
  // Constructor
  template< /*class F,*/ int Dim >
  Alps::SubjectCPU<Dim>::SubjectCPU( const int SubNumber,
				      const std::size_t NumModalities ): /*Alps::Subject(),*/
    subject_number_{SubNumber}, number_modalities_{NumModalities}
  {}
  //
  // 
  template< /*class F,*/ int Dim > void
  Alps::SubjectCPU<Dim>::add_modalities( const std::string Modality )
  {
    try
      {
	if ( Alps::file_exists(Modality) )
	  {
	    std::cout << Modality << std::endl;
	    //
	    // load the image ITK pointer
	    auto image_ptr = itk::ImageIOFactory::CreateImageIO( Modality.c_str(),
								 itk::CommonEnums::IOFileMode::ReadMode );
	    image_ptr->SetFileName( Modality );
	    image_ptr->ReadImageInformation();
	    // Read the ITK image
	    typename Reader<Dim>::Pointer img_ptr = Reader<Dim>::New();
	    img_ptr->SetFileName( image_ptr->GetFileName() );
	    img_ptr->Update();
	    //
	    modalities_.push_back( img_ptr->GetOutput() );
	    modality_size_.push_back( img_ptr->GetOutput()->GetLargestPossibleRegion().GetSize() );
	  }
	else
	  {
	    std::string mess = "Image (";
	    mess += Modality + ") does not exists.";
	    throw MAC::MACException( __FILE__, __LINE__,
				     mess.c_str(),
				     ITK_LOCATION );
	  }
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
      }
  }
}
#endif
