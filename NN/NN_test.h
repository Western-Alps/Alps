#ifndef NN_TEST_H
#define NN_TEST_H
//
//
//
#include <iostream>
#include <fstream>
#include <math.h>  
#include <map>
#include <list>
#include <memory>
//
// CUDA
//
#include <cuda_runtime.h>
//
//
//
#include "MACException.h"
#include "Subject.h"
#include "NeuralNetwork.h"
//
//
//
namespace MAC
{

  /** \class NN_test
   *
   * \brief 
   * 
   * 
   */
  class NN_test : public NeuralNetwork
    {
      //
      // Some typedef
      using Image3DType = itk::Image< double, 3 >;
      using Reader3D    = itk::ImageFileReader< Image3DType >;

    public:
      /** Constructor. */
      NN_test();
      //
      //explicit Subject( const int, const int );

      /** Destructor */
      virtual ~NN_test(){};

      //
      // Initialization
      virtual void initialization(){};
      //
      // get the layer name
      virtual std::string get_layer_name(){ return std::string("test layer");};
      //
      // get the layer name
      virtual Layer get_layer_type(){ return neural_network_test_class;};
      //
      // get the layer name
      virtual double get_energy(){ return 0.;};
      //
      // Forward propagation
      virtual void forward( Subject&, const Weights& W = Weights() );
      //
      //
      virtual void backward(){};
      //
      // Backward error propagation
      virtual void backward_error_propagation(){};
      //
      //
      virtual void add( std::shared_ptr< NeuralNetwork > ){};
      //
      //
      virtual int get_number_weights() const { return 1;};

    private:
      //
      // Weights
      double* weights_;
      double* d_weights_;
    };
}
#endif
