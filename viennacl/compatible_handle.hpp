/* 
 * File:   compatible_handle.hpp
 * Author: bsp
 *
 * Created on 9 September 2015, 7:27 PM
 */

#ifndef COMPATIBLE_HANDLE_HPP
#define	COMPATIBLE_HANDLE_HPP

#include "viennacl/forwards.h"
#include "viennacl/tools/shared_ptr.hpp"

#ifdef VIENNACL_WITH_OPENCL
#include "CL/cl.h"
#include "viennacl/ocl/forwards.h"
#endif



namespace viennacl
{

  
  class compatible_handle
  {
  public:
    virtual ~compatible_handle(){};

  };


 #ifdef VIENNACL_WITH_HSA
 class hsa_compatible_handle 
  {
  public:

  typedef viennacl::tools::shared_ptr<char>      ram_handle_type;

  /** @brief Returns the handle to a buffer in CPU RAM. NULL is returned if no such buffer has been allocated. */
  virtual ram_handle_type const & hsa_handle() const =0;
 };
#endif
 
#ifdef VIENNACL_WITH_OPENCL      
  class opencl_compatible_handle
  {
  public:
/** @brief Returns the handle to an OpenCL buffer. The handle contains NULL if no such buffer has been allocated. */
  virtual viennacl::ocl::handle<cl_mem> const & opencl_handle() const = 0;
    
  };
#endif  
}


#endif	/* COMPATIBLE_HANDLE_HPP */

