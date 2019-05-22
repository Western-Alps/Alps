#ifndef CONVOLUTIONAL_WINDOW_H
#define CONVOLUTIONAL_WINDOW_H
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
#include "Weights.h"
//#include "NeuralNetwork.h"
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
  /*! \class Convolutional_window
   * \brief class representing the shared weights in a convolutional layer.
   *
   */
  class Convolutional_window  : public Weights  
  {
  public:
    //
    // Constructor
    Convolutional_window();
    //
    // Constructor
    Convolutional_window( const std::string );
    //
    // Destructor
    ~Convolutional_window(){};
    
    
  public:
    //
    // Accessors
    virtual void print();
    // Save the weightd
    virtual void save_weights(){};
    // Save the weightd
    virtual void load_weights(){};

  private:
    int stride_[3]{0,0,0};
    int padding_[3]{0,0,0};
    // Half window
    int half_window_[3]{0,0,0};
    //
    // Outputs
    // Number of features: 2,4,8,16,...,1024
    int number_of_features_{0};
  };
}
#endif
