#ifndef ALPSSUBJECT_H
#define ALPSSUBJECT_H
//
//
//
#include <iostream> 
//
// 
//
#include "MACException.h"
#include "AlpsClimber.h"
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
  //template< /*class Function,*/ int Dim >
  class Subject
  {
  public:

    //
    /** Constructor */
    //explicit Subject();
    /* Destructor */
    virtual ~Subject(){};

    //
    // Accessors


    //
    // functions
    virtual void add_modalities( const std::string ) = 0;
    virtual bool check_modalities() const            = 0;
  };
}
#endif
