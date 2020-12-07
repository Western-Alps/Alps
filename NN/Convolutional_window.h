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
#include <memory>
#include <random>
//
// Eigen
//
#include <Eigen/Sparse>
using IOWeights = Eigen::Triplet< double >;
// ITK
#include "ITKHeaders.h"
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
    explicit Convolutional_window();
    //
    // Constructor
    // WARNING: This constructor is targetting the first
    // convolution. Subsequent window must take a window as
    // input.
    explicit Convolutional_window( const std::string, std::shared_ptr< Convolutional_window >,
				   const int*, const int*, const int*,
				   const int );
    //
    // Constructor
    // WARNING: This constructor is for subsequent window.
    // It comes after a first convolutional window.
    explicit Convolutional_window( const std::string,
				   const int*, const int*, const int*,
				   const int );
    //
    // Destructor
    virtual ~Convolutional_window();
    
    
  public:
    //
    // Functions
    virtual void print();
    // Check we have the right image as input
    void         check_match( ImageType<3>::SizeType,
			       ImageType<3>::SizeType);
    //
    // Accessors
    //
    // Save the weights
    virtual void               save_weights();
    // Save the weights
    virtual void               load_weights();
    // type of Window
    virtual inline Weight_type get_layer_type(){ return Conv_layer; };
    //
    // from the class
    const std::size_t get_number_of_features_in()  const { return number_of_features_in_;};
    const std::size_t get_number_of_features_out() const { return number_of_features_out_;};
    // Image information input layer
    const ImageType<3>::SizeType      get_size_in()      const { return size_in_; };
    const ImageType<3>::SpacingType   get_spacing_in()   const { return spacing_in_; };
    const ImageType<3>::PointType     get_origine_in()   const { return origine_in_; };
    const ImageType<3>::DirectionType get_direction_in() const { return direction_in_; };
    // Image information output layer
    const ImageType<3>::SizeType      get_size_out()      const { return size_out_; };
    const ImageType<3>::SpacingType   get_spacing_out()   const { return spacing_out_; };
    const ImageType<3>::PointType     get_origine_out()   const { return origine_out_; };
    const ImageType<3>::DirectionType get_direction_out() const { return direction_out_; };
    //
    std::size_t                      get_im_size_in() const { return im_size_in_; };
    std::size_t                      get_im_size_out() const { return im_size_out_; };
    std::size_t**                    get_weights_position_oi() const { return weights_poisition_oi_; };
    std::size_t**                    get_weights_position_io() const { return weights_poisition_io_; };
    //
    const Eigen::SparseMatrix< std::size_t >& get_W_out_in() const {return W_out_in_; };
    const std::vector< IOWeights >            get_triplet_oiw() const {return triplet_oiw_; };
    //
    std::map< std::string, std::tuple<
      std::vector< std::shared_ptr<double> > /* activations */,
      std::vector< std::shared_ptr<double> > /* neurons */,
      std::vector< std::shared_ptr<double> > /* deltas */ > >& get_neuron(){ return neurons_;};
    //
    std::shared_ptr< Convolutional_window > get_previouse_conv_window() {return previouse_conv_window_;};

    
  private:
    // This function creates the size of the output feature 
    // following each direction Dim.
    ImageType<3>::SizeType feature_size( const ImageType<3>::SizeType ) const;
    // This function creates the size of the output feature 
    // following each direction Dim.
    ImageType<3>::PointType feature_orig( const ImageType<3>::SizeType,
					 const ImageType<3>::SpacingType,
					 const ImageType<3>::PointType ) const;

    //
    // Neurons, activations and delta
    std::map< std::string, std::tuple<
      std::vector< std::shared_ptr<double> > /* activations */,
      std::vector< std::shared_ptr<double> > /* neurons */,
      std::vector< std::shared_ptr<double> > /* deltas */ > > neurons_;

    // 
    // Inputs
    std::shared_ptr< Convolutional_window > previouse_conv_window_;
    // Half window
    int*                                    convolution_half_window_size_{nullptr};
    int*                                    stride_{nullptr};
    int*                                    padding_{nullptr};

    //
    // Outputs
    // Number of features: 2,4,8,16,...,1024
    std::size_t                number_of_features_in_;
    std::size_t                number_of_features_out_;
    // Image information input layer
    ImageType<3>::SizeType      size_in_;
    ImageType<3>::SpacingType   spacing_in_;
    ImageType<3>::PointType     origine_in_;
    ImageType<3>::DirectionType direction_in_;
    // Image information output layer
    ImageType<3>::SizeType      size_out_;
    ImageType<3>::SpacingType   spacing_out_;
    ImageType<3>::PointType     origine_out_;
    ImageType<3>::DirectionType direction_out_;
    // Weights position in the Weight matrix
    std::size_t                im_size_in_{0};
    std::size_t                im_size_out_{0};
    std::size_t**              weights_poisition_oi_{nullptr};
    std::size_t**              weights_poisition_io_{nullptr};
    // Complete matrix
    Eigen::SparseMatrix< std::size_t > W_out_in_;
    std::vector< IOWeights >           triplet_oiw_;
  };
}
#endif
