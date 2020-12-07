#ifndef ALPSSUBJECT_H
#define ALPSSUBJECT_H
//
//
//
#include <iostream> 
#include <memory>
//
// 
//
#include "MACException.h"
#include "AlpsClimber.h"
#include "AlpsMountain.h"
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
  /*! \class Subject
   *
   * \brief class Subject record the information 
   * of the subject through the processing.
   *
   */
  template< /*class Function,*/ int Dim >
  class Subject
  {
  public:

    //
    /** Constructor */
    explicit Subject();
    /* Destructor */
    virtual ~Subject(){};

    //
    // Accessors


    //
    // functions
    virtual void add_modalities( const std::string );
    virtual bool check_modalities() const { number_modalities_ == modalities_.size() ? true : false;};

  private:
    // 
    //std::vector<  > modalities_;
    // This function is the continuous step function
    Function activation_function_;
  };
  
  //
  // Constructor
  template< /*class F,*/ class A, int Dim >
  Alps::Subjects<F,A,Dim>::Subjects():
  {
  }
  //
  // 
  template< /*class F,*/ int Dim > void
  Alps::Subjects<F,A,Dim>::update( std::shared_ptr< Alps::Mountain > Mount )
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
