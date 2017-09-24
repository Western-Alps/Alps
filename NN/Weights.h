#ifndef WEIGHTS_H
#define WEIGHTS_H
//
//
//
#include <iostream> 
#include <fstream>
#include <string>
#include <vector>
//
// 
//
#include "MACException.h"
//
//
//
/*! \namespace MAC
 *
 * Name space for MAC.
 *
 */
namespace MAC
{
  /*! \class Weights
   * \brief class representing the shared weight between all the neural networks layers.
   *
   */
  class Weights
  {
  public:
    //
    // Constructor
    Weights();
    //
    // Constructor
    Weights( const int , const std::vector< int > );

    //
    // Accessors
    const double* get_weights() const { return weights_; }
    const std::vector< int >& get_weight_indexes() const { return weight_indexes_; }

  public:
    //
    // Accessors
    void print();
    
    //
//    // Single instance
//    static Weights* weights_instance()
//    {
//      if ( !weights_instance_ )
//	weights_instance_ = new Weights();
//      return weights_instance_;
//    }


  private:
    // Number of weights
    int                 number_of_weights_{0};
    // The weight indexes is an array of indexes where starts the weights of this layer
    std::vector< int >  weight_indexes_;
    // weights
    double*             weights_;
//    //! Unique instance
//    static Weights *weights_instance_;
  };
}
#endif
