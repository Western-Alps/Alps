#ifndef ALPSTENSOR_H
#define ALPSTENSOR_H
//
//
//
#include <iostream>
#include <cassert>
#include <string>
#include "MACException.h"
//
//
//
namespace Alps
{
  /** \class Tensor
   *
   * \brief 
   * Tensor object represents the basic memory element for the neural network.
   * 
   */
  template< class Type, int Dim >
  class Tensor
    {
      //
      // 
    public:
      /** Constructor. */
      explicit Tensor();
    
      /** Destructor */
      virtual ~Tensor(){};

      //
      // Accessors
      inline const int get_dimension() const {return Dim;}
      const int get_size( const int S ) const
      {
	assert( S < Dim );
	return size_[S];
      }
      
    private:
      //
      // private member function
      //

      //! Member representing the size along each dimension.
      int* size_{nullptr};
    };
  //
  //
  //
  template< class T, int D>
    Tensor< T, D >::Tensor()
  {
    try
      {
      if ( D == 0 )
	{
	  std::string mess = "Tensor class does not handle tensor 0 yet.";
	  throw MAC::MACException( __FILE__, __LINE__,
				   mess.c_str(),
				   ITK_LOCATION );
	}
      //
      size_ = (int*) malloc( D * sizeof(int) );
      for ( int d = 0 ; d < D ; d++ )
	size_[d] = 0;
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit(-1);
      }
  }
}
#endif
