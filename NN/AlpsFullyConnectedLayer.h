#ifndef ALPSFULLYCONNECTEDLAYER_H
#define ALPSFULLYCONNECTEDLAYER_H
//
//
//
#include <iostream>
#include <memory>
//
#include "MACException.h"
#include "AlpsLayer.h"
#include "AlpsSubjects.h"
#include "AlpsMountain.h"
//#include "AlpsLayerDependencies.h"
//#include "AlpsWeights.h"
//
//
//
namespace Alps
{
  /** \class FullyConnectedLayer
   *
   * \brief 
   * FullyConnectedLayer class represents the basic layer element that can be used 
   * into a densly connected neural network.
   * 
   */
  template< typename ActivationFunction, int Architecture, int Dim  >
  class FullyConnectedLayer : public Alps::Layer, public Alps::Mountain
  {
    //
    // 
  public:
    /** Constructor. */
    explicit FullyConnectedLayer( const std::string, const int,
				    const int,         const int* );
    
    /** Destructor */
    virtual ~FullyConnectedLayer(){};

    //
    // Accessors

    // Forward propagation
    virtual void forward()                                    override {};
    // Backward propagation
    virtual void backward()                                   override {};
    // Attach observers that need to be updated
    virtual void attach( std::shared_ptr< Alps::Climber > )   override;
    // Notify the observers for updates
    virtual void notify()                                     override {};
//    // Update the weights
//    virtual void update_weights() override {};
//    // Update the weights
//    virtual void attach_weights( std::shared_ptr< Weights > Weights ) override
//    { weights_ = Weights; };

  private:
    //
    // private member function
    //

      
    //
    // Convolutional layer's name
    std::string layer_name_;
    // layer energy
    double      energy_{0.};
    // Weights
    const int   layer_number_;
    // number of fully connected layers
    const int   number_fc_layers_;
    // 
    int*        fc_layers_;

    //
    // Observers
    // Observers conatainer
    std::list< std::shared_ptr< Alps::Climber > > climbers_;
//    // Subjects
//    std::shared_ptr< Alps::Subjects< /*ActivationFunction,*/ Architecture, Dim > > subjects_;
  };
  //
  //
  template< class AF, int A, int D   >
  FullyConnectedLayer< AF, A, D >::FullyConnectedLayer( const std::string Layer_name,
							  const int         Layer_number,
							  const int         Number_fc_layers,
							  const int*        Fc_layers ):
    layer_name_{Layer_name}, layer_number_{Layer_number}, number_fc_layers_{Number_fc_layers},
    fc_layers_{ new int[Number_fc_layers] }
  {
    try
      {
	//
	//
	memcpy( fc_layers_, Fc_layers, Number_fc_layers*sizeof(int) );

	//
	// Create the subjects (images)
	std::shared_ptr< Alps::Subjects< /*ActivationFunction,*/ A, D > >
	  subjects = std::make_shared< Alps::Subjects< /*AF,*/ A, D > >( std::shared_ptr< FullyConnectedLayer< AF, A, D > >( this ) );
	// Attached the subjects
	attach( subjects );

  
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit(-1);
      }
  };
  //
  //
  template< class AF, int A, int D   > void
  FullyConnectedLayer< AF, A, D >::attach( std::shared_ptr< Alps::Climber > One_climber )
  {
    try
      {
	climbers_.push_back( One_climber );
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit(-1);
      }
  };
}
#endif
