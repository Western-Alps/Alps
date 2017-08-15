#ifndef MACEXCEPTION_H
#define MACEXCEPTION_H
#include "itkMacro.h"
#include "itkExceptionObject.h"
//
//
//
namespace MAC
{
  /** \class MACException
   *
   * \brief Base exception class for classification conflicts.
   * 
   */
  class MACException : public itk::ExceptionObject
  {
  public:
    /** Run-time information. */
    itkTypeMacro(ImageFileReaderException, ExceptionObject);
    /** Constructor. */
  MACException( const char *file, unsigned int line,
		const char *message = "Error in Bmle",
		const char *loc = "Unknown" ):
    ExceptionObject( file, line, message, loc ){}
    /** Constructor. */
  MACException( const std::string & file, unsigned int line,
		const char *message = "Error in Bmle",
		const char *loc = "Unknown" ):
    ExceptionObject( file, line, message, loc ){}
    /** Has to have empty throw(). */
    virtual ~MACException() throw() {};
  };
}
#endif
