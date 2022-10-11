#ifndef ALPSBASEFUNCTION_H
#define ALPSBASEFUNCTION_H
//
//
//
namespace Alps
{
  enum class Func
    {
     UNKNOWN   = -1,
     // Activation functions
     F_TANH    = 1,
     F_SIGMOID = 2,
     F_RELU    = 3,
     F_LINEAR  = 4,
    // Cost functions
     L_LSE     = 100
    };
  /** \class BaseFunction
   *
   * \brief 
   * This class is the base class for all the functions.
   * 
   */
  class BaseFunction
    {
    public:
      /** Destructor */
      virtual ~BaseFunction(){};


      //
      // Accessors
      //
      // get function name 
      virtual Func get_function_name() const = 0;


      //
      // Functions
      //
    };
}
#endif
