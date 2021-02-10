#ifndef ALPSWEIGHTSFCL_H
#define ALPSWEIGHTSFCL_H
//
//
//
#include <iostream> 
#include <vector>
#include <memory>
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
//
#include "MACException.h"
#include "AlpsWeights.h"
#include "AlpsClimber.h"
#include "AlpsMountain.h"
//
//
//
/*! \namespace Alps
 *
 * Name space for Alps.
 *
 */
namespace Alps
{
  /*! \class WeightsFullyConnected
   * \brief class represents the weights container for fully
   * connected layers (FCL).
   *
   */
  template< typename Tensor1_Type,
	    typename Tensor2_Type,
	    Alps::Arch Architecture,
	    typename Activation,
	    typename Solver >
  class WeightsFcl : public Alps::Weights< Tensor1_Type, Tensor2_Type >,
		     public Alps::Climber
  {
  public:
    /** Constructor. */
    explicit WeightsFcl( std::shared_ptr< Alps::Mountain >,
			 const std::vector< std::size_t >,
			 const std::vector< std::size_t > ){};
    /** Destructor */
    virtual ~WeightsFcl(){};


    //
    // Accessors
    //
    // Activation tensor from the previous layer
    virtual void set_previous_activation()                                                                    override{};
    // Current derivative of the activation
    virtual void set_current_dactivation()                                                                    override{};
    // Weights from the next layer
    virtual void set_next_weights()                                                                           override{};
    // Get size of the tensor
    virtual const std::vector< std::size_t >   get_tensor_size() const                                override
    { return std::vector< std::size_t >(); };						      
    // Get the tensor			     						      
    virtual std::shared_ptr< Tensor2_Type >    get_tensor() const                                     override
    { return weights_;};			     						      
    // Set size of the tensor		     						      
    virtual void                               set_tensor_size( std::vector< std::size_t > )          override{};
    // Set the tensor			     						      
    virtual void                               set_tensor( std::shared_ptr< Tensor2_Type > )          override{};
    //					     						      
    //					     						      
    // Get the observed mountain	     						      
    virtual std::shared_ptr< Alps::Mountain >  get_mountain()                                         override
    { return layer_;};
    // Get energy
    virtual const double                       get_energy() const                                     override
    { return 0.;};

    												      
    //												      
    // Functions										      
    //												      
    // Save the weights										      
    virtual void                               save_tensor() const                                    override{};
    // Load the weights										      
    virtual void                               load_tensor( const std::string )                       override{};
    //
    //
    // Activate
    virtual std::tuple < std::shared_ptr< Tensor1_Type >,
			 std::shared_ptr< Tensor1_Type > > activate( std::vector< Alps::LayerTensors< Tensor1_Type, 2 > >& ) override{};
    // solver
    virtual void                                           solve()                                    override{};
    //
    //
    // Update the weights
    virtual void                               update()                                               override{};



  private:
    // Matrix of weigths
    std::shared_ptr< Tensor2_Type >   weights_{nullptr};
    // weights activation
    Activation                        activation_;
    //
    // The mountain observed: fully connected layer
    std::shared_ptr< Alps::Mountain > layer_{nullptr};
  };
  /*! \class WeightsFullyConnected
   * \brief class representing the weights container for fully
   * connected layers (FCL) using CPU.
   *
   */
  template< typename Type,
	    typename Activation,
	    typename Solver >
  class WeightsFcl< Type, Eigen::MatrixXd, Alps::Arch::CPU, Activation, Solver > : public Alps::Weights< Type, Eigen::MatrixXd >,
										      public Alps::Climber
  {
  public:
    /** Constructor. */
    explicit WeightsFcl( std::shared_ptr< Alps::Mountain >,
			 const std::vector< std::size_t >,
			 const std::vector< std::size_t > );
    /** Destructor */
    virtual ~WeightsFcl(){};


    //
    // Accessors
    //
    // Activation tensor from the previous layer
    virtual void set_previous_activation() override{};
    // Current derivative of the activation
    virtual void set_current_dactivation() override{};
    // Weights from the next layer
    virtual void set_next_weights() override{};
    // Get size of the tensor
    virtual const std::vector< std::size_t >      get_tensor_size() const                             override
    { return std::vector< std::size_t >(); };							      
    // Get the tensor										      
    virtual std::shared_ptr< Eigen::MatrixXd >    get_tensor() const                                  override
    {return weights_;};										      
    // Set size of the tensor									      
    virtual void                                  set_tensor_size( std::vector< std::size_t > )       override{};
    // Set the tensor										      
    virtual void                                  set_tensor( std::shared_ptr< Eigen::MatrixXd > )    override{};
    //												      
    //												      
    // Get the observed mountain								      
    virtual std::shared_ptr< Alps::Mountain >     get_mountain()                                      override
    {return layer_;};										      
    // Get energy
    virtual const double                          get_energy() const                                  override
    { return 0.;};

    
    //												      
    // Functions										      
    //												      
    // Save the weights										      
    virtual void                                  save_tensor() const                                 override{};
    // Load the weights										      
    virtual void                                  load_tensor( const std::string )                    override{};
    //
    //
    // Activate
    virtual std::tuple < std::shared_ptr< Type >,
			 std::shared_ptr< Type > > activate( std::vector< Alps::LayerTensors< Type, 2 > >& ) override;
    // solver
    virtual void                                   solve()                                            override{};



    //
    //
    // Update the weights
    virtual void                                   update()                                            override{};



  private:
    // Matrix of weigths
    std::shared_ptr< Eigen::MatrixXd > weights_;
    // weights activation
    Activation                         activation_;
    // input image from the previous layer
    Eigen::MatrixXd                    z_in_;
    //
    // The mountain observed: fully connected layer
    std::shared_ptr< Alps::Mountain >  layer_;
  };
  //
  //
  //
  template< typename T, typename A, typename S >
  Alps::WeightsFcl< T, Eigen::MatrixXd, Alps::Arch::CPU, A, S >::WeightsFcl( std::shared_ptr< Alps::Mountain > Layer,
									     const std::vector< std::size_t >  Layer_size,
									     const std::vector< std::size_t >  Prev_layer_size  ):
    layer_{Layer}
  {
    try
      {
	//
	// Get the number of layers defined
	std::size_t
	  prev_num_of_layers    = Prev_layer_size.size(),
	  current_num_of_layers = Layer_size.size();
	//
	//
	int
	  current_nodes  = 0,
	  previous_nodes = 0;
	// How many nodes we had in this layer:
	for ( std::size_t l = 0 ; l < current_num_of_layers ; l++ )
	  current_nodes += Layer_size[l];
	// How many nodes we had in the previous layer:
	for ( std::size_t pl = 0 ; pl < prev_num_of_layers ; pl++ )
	  previous_nodes += Prev_layer_size[pl];
	// Random create the variables between [-1,1]
	weights_ = std::make_shared< Eigen::MatrixXd >(Eigen::MatrixXd::Random( current_nodes,
										previous_nodes + 1 /* biais */) );
	std::cout
	  << "Weights ["<<weights_->rows()<<"x"<<weights_->cols()<<"]" 
	  << std::endl;
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
      }
  };
  //
  //
  //
  template< typename T, typename A, typename S > std::tuple< std::shared_ptr< T >,
							     std::shared_ptr< T > >
  Alps::WeightsFcl< T, Eigen::MatrixXd, Alps::Arch::CPU, A, S >::activate( std::vector< Alps::LayerTensors< T, 2 > >& Image_tensors )
  {
    //
    // Check the dimensions are right
    long int tensors_size = 0;
    for ( auto tensor : Image_tensors )
      tensors_size += static_cast< long int >( tensor.get_tensor_size()[0] );
    //
    if ( weights_->cols() != tensors_size + 1 /*bias*/ )
      {
	std::string
	  mess = std::string("There is miss match between the number of columns (")
	  + std::to_string( weights_->cols() )
	  + std::string(") and the size of the input tensor (")
	  + std::to_string( tensors_size )
	  + std::string("+1).");
	throw MAC::MACException( __FILE__, __LINE__,
				 mess.c_str(),
				 ITK_LOCATION );
      }
    //
    // reset the z_in_ and save the information in the object
    z_in_ = Eigen::MatrixXd::Zero( weights_->cols(), 1 );
    // Converter the tensor into an Eigen matrix
    std::shared_ptr< T > z_out  = std::shared_ptr< T >( new  T[weights_->rows()],
							std::default_delete< T[] >() );
    std::shared_ptr< T > dz_out = std::shared_ptr< T >( new  T[weights_->rows()],
							std::default_delete< T[] >() );
    Eigen::MatrixXd      a_out  = Eigen::MatrixXd::Zero( weights_->rows(), 1 );
    // Load the tensor image into a Eigen vector
    std::size_t shift = 1;
    z_in_(0,0) = 1.; // bias
    for ( auto tensor : Image_tensors )
      {
	std::size_t img_size = tensor.get_tensor_size()[0];
	for ( std::size_t s = 0 ; s < img_size ; s++ )
	  z_in_(s+shift,0) = tensor[TensorOrder1::ACTIVATION][s];
	shift += img_size;
      }
    // process
    a_out = *( weights_.get() ) * z_in_;
    // Apply the activation function
    long int activation_size = weights_->rows();
    for ( long int s = 0 ; s < activation_size ; s++ )
      {
	z_out.get()[s]  = activation_.f( a_out(s,0) );
	dz_out.get()[s] = activation_.df( a_out(s,0) );
      }

    //
    //
    return std::make_tuple( z_out, dz_out );
  };
  /*! \class WeightsFullyConnected
   * \brief class representing the weights container for fully
   * connected layers (FCL) using GPU.
   *
   */
  template< typename Type1,
	    typename Type2,
	    typename Activation,
	    typename Solver >
  class WeightsFcl< Type1, Type2, Alps::Arch::GPU, Activation, Solver > : public Alps::Weights< Type1, Type2 >,
									  public Alps::Climber
  {
  public:
    /** Constructor. */
    explicit WeightsFcl( std::shared_ptr< Alps::Mountain >,
			 const std::vector< std::size_t >,
			 const std::vector< std::size_t > ){std::cout << "GPU treatment" << std::endl;};
    /** Destructor */
    virtual ~WeightsFcl(){};


    //
    // Accessors
    //
    // Activation tensor from the previous layer
    virtual void set_previous_activation()                                                                     override{};
    // Current derivative of the activation
    virtual void set_current_dactivation()                                                                     override{};
    // Weights from the next layer
    virtual void set_next_weights()                                                                            override{};
    // Get size of the tensor
    virtual const std::vector< std::size_t >      get_tensor_size() const                              override
    { return std::vector< std::size_t >(); };							      
    // Get the tensor										      
    virtual std::shared_ptr< Type2 >              get_tensor() const                                   override
    {return weights_;};										      
    // Set size of the tensor									      
    virtual void                                  set_tensor_size( std::vector< std::size_t > )        override{};
    // Set the tensor										      
    virtual void                                  set_tensor( std::shared_ptr< Type2 > )               override{};
    //												      
    //												      
    // Get the observed mountain								      
    virtual std::shared_ptr< Alps::Mountain >     get_mountain()                                       override
    {return layer_;};										      
    // Get energy
    virtual const double                          get_energy() const                                   override
    { return 0.;};
												      
    												      
    //												      
    // Functions										      
    //												      
    // Save the weights										      
    virtual void                            save_tensor() const                                        override{};
    // Load the weights										      
    virtual void                            load_tensor( const std::string )                           override{};
    //
    //
    // Activate
    virtual std::tuple < std::shared_ptr< Type1 >,
			 std::shared_ptr< Type1 > > activate( std::vector< Alps::LayerTensors< Type1, 2 > >& ) override{};
    // solver
    virtual void                                    solve()                                            override{};



    //
    //
    // Update the weights
    virtual void                            update()                                                   override{};



  private:
    // Matrix of weigths
    std::shared_ptr< Type2 >           weights_;
    // weights activation
    Activation                         activation_;
    //
    // The mountain observed: fully connected layer
    std::shared_ptr< Alps::Mountain >  layer_;
  };
}
#endif
