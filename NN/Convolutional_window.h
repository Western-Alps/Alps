#ifndef CONVOLUTIONAL_WINDOW_H
#define CONVOLUTIONAL_WINDOW_H
//
//
//
#include <iostream> 
#include <fstream>
#include <cstring>
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
   * convolution_half_window_size_: 
   * 	The window size is a 3 dimensions, the dimension must be odd! because we are
   * 	taking in account the center of the convolution.
   * 	3 dimensions for x,y and z; 
   * 	Note: To complete the kernel size, we need to know how many feature maps we 
   *    had in the previouse round.
   *
   * stride_:
   * 	The stride array represents the jump of a kernel center for the next convolution.
   *
   * padding_:
   * 	ToDo: This feature is ignored for the first version of the software.
   * 
   * transpose_:
   * The convolution is used when the kernel is direct (transpose = false). The 
   * deconvolution is the reverse operation (transpose = true).
   * ToDo: the transpose operation needs to make sure a convolution was done 
   *       before.
   *
   *  number_of_features_:  
   * 	Number of features: 2,4,8,16,...,1024. Represents the number of kernel we
   *    are going to build.
   *
   *
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
    Convolutional_window( const std::string,
			  const int*, const int*, const int*,
			  const int );
    //
    // Destructor
    virtual ~Convolutional_window();
    
    
  public:
    //
    // Accessors
    virtual void print();
    // Save the weightd
    virtual void save_weights(){};
    // Save the weightd
    virtual void load_weights(){};

    //
    // Transpose the weights in the process of
    // convolution <-> deconvolution
    void   transpose(){ transpose_ = true;};
    // Output feature size
    // This function creates the size of the output feature 
    // following each direction Dim.
    int feature_size( const int ) const;

  private:
    // 
    // Inputs
    // Half window
    int convolution_half_window_size_[3]{0,0,0};
    int stride_[3]{0,0,0};
    int padding_[3]{0,0,0};

    //
    // Operations
    bool transpose_{false};

    //
    // Outputs
    // Number of features: 2,4,8,16,...,1024
    int number_of_features_;
  };
}
#endif
