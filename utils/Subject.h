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
// ITK
#include "ITKHeaders.h"
//
//
// Alps
#include "MACException.h"
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
      // 
    public:
      /** Constructor. */
      explicit Subject():
      label_{0}{};
    
      /** Destructor */
      virtual ~Subject(){};

      //
      // Accessors
      const std::vector< ImageType<3>::Pointer >&  get_modalities_ITK_images() const
      {
	return modalities_ITK_images_;
      }
      const std::vector< ImageType<3>::Pointer >&  get_modality_targets_ITK_images() const
      {
	return modality_targets_ITK_images_;
      }
      const std::vector< ImageType<3>::SizeType >& get_modality_images_size() const
      {
	return modality_images_size_;
      }
      const std::vector< ImageType<3>::Pointer >&  get_clone_modalities_images() const
      {
	return clone_modalities_images_;
      }
      std::string                                 get_subject_name() const
	{
	  return name_;
	}
      int                                         get_subject_label() const
	{
	  return label_;
	}
      //
      //
      void set_subject_name( std::string Name )
      {
	name_ = Name;
      }
      void set_clone_modalities_images( std::vector< ImageType<3>::Pointer >& Imgs )
      {
	clone_modalities_images_ = Imgs;
      }

      //
      // 
      void write() const;
      //
      // 
      void write_clone() const;

      //
      // Add modality
      void add_modality( const std::string );

      //
      // Add a target for the modality
      void add_modality_target( const std::string );
      //
      // Add label
      void add_label( const int Label )
      {
	label_ = Label;
      };
      //
      // Update the current read image
      void update();
      //
      // Update the current read image
      void update( const std::vector< ImageType<3>::Pointer >  );

    private:
      //
      // private member function
      //

      //
      // Subject parameters
      //

      // subject name
      std::string name_;
      // vector of modalities
      std::vector< ImageType<3>::Pointer > modalities_ITK_images_;
      // vector of targets
      std::vector< ImageType<3>::Pointer > modality_targets_ITK_images_;
      // Current read images
      // This set of images will be transfered to next neural network layers
      std::vector< ImageType<3>::Pointer > clone_modalities_images_;
      // images size
      std::vector< ImageType<3>::SizeType > modality_images_size_;
      // label
      int label_;
    };
}
#endif
