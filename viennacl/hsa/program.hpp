#ifndef VIENNACL_HSA_PROGRAM_HPP_
#define VIENNACL_HSA_PROGRAM_HPP_

/* =========================================================================
   Copyright (c) 2010-2014, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/** @file viennacl/hsa/program.hpp
    @brief Implements an OpenCL program class for ViennaCL
*/

#include <string>
#include <vector>
#include "viennacl/hsa/forwards.h"
#include "viennacl/hsa/brig_helper.hpp"
#include "viennacl/hsa/handle.hpp"
#include "viennacl/hsa/kernel.hpp"
#include "viennacl/tools/shared_ptr.hpp"


namespace viennacl
{
namespace hsa
{




/** @brief Wrapper class for an OpenCL program.
  *
  * This class was written when the OpenCL C++ bindings haven't been standardized yet.
  * Regardless, it takes care about some additional details and is supposed to provide higher convenience by holding the kernels defined in the program.
  */
class program
{
  typedef std::vector<tools::shared_ptr<viennacl::hsa::kernel> >    kernel_container_type;

public:
  program() : p_context_(NULL) {}
  program(brig_module program_module, viennacl::hsa::context const & program_context, std::string const & prog_name = std::string())
    : handle_(program_module, program_context), p_context_(&program_context), name_(prog_name)
  {
  }

  program(program const & other) : handle_(other.handle_), p_context_(other.p_context_), name_(other.name_), kernels_(other.kernels_) {      }

  viennacl::hsa::program & operator=(const program & other)
  {
    handle_ = other.handle_;
    name_ = other.name_;
    p_context_ = other.p_context_;
    kernels_ = other.kernels_;
    return *this;
  }

  viennacl::hsa::context const * p_context() const { return p_context_; }

  std::string const & name() const { return name_; }

  /** @brief Returns the kernel with the provided name */
  inline viennacl::hsa::kernel & get_kernel(std::string const & name);    //see context.hpp for implementation

  const viennacl::hsa::handle<brig_module> & handle() const { return handle_; }

  hsa_status_t finalize(hsa_agent_t device)
  {
	  hsa_status_t status = HSA_STATUS_SUCCESS;
      hsa_ext_program_handle_t hsa_program;
      hsa_program.handle = 0;
	    //Create hsa program.
	    status = hsa_ext_program_create(&device, 1, HSA_EXT_BRIG_MACHINE_LARGE, HSA_EXT_BRIG_PROFILE_FULL, &hsa_program);

	    //Add BRIG module to hsa program.
	    hsa_ext_brig_module_handle_t module;
	    status = hsa_ext_add_module(hsa_program, handle_.get().brig_module_, &module);
	    // entry offset into the code section.
	    std::vector<hsa_ext_finalization_request_t> finalization_request_list;
	    std::for_each(handle_.get().kernels_.begin(), handle_.get().kernels_.end(), [&finalization_request_list, &module](const kernel_entry& entry){
	    	hsa_ext_finalization_request_t request;
	    	memset(&request, 0, sizeof(hsa_ext_finalization_request_t));
	    	request.module = module;              // module handle.
	    	request.program_call_convention = 0;  // program call convention. not supported.
	    	request.symbol = entry.offset_;
	    	finalization_request_list.push_back(request);
	    } );

	    //Finalize hsa program.
	    status = hsa_ext_finalize_program(hsa_program, device, finalization_request_list.size(), &finalization_request_list[0], NULL, NULL, 0, NULL, 0);


	    hsa_region_t region;
    	hsa_agent_iterate_regions(device,program::get_kernarg, &region);

	    for (size_t i = 0 ;i < finalization_request_list.size() ; ++i)
	    {
	    	// create kernargs
		    hsa_ext_code_descriptor_t *hsa_code_descriptor;
		    hsa_ext_query_kernel_descriptor_address(hsa_program, module, finalization_request_list[i].symbol, &hsa_code_descriptor);
	    	size_t run_kernel_arg_buffer_size =  hsa_code_descriptor->kernarg_segment_byte_size;
	    	kernel_arg_buffer buffer(region, run_kernel_arg_buffer_size,  handle_.get().kernels_[i].arg_count_ );
	    	kernels_.push_back(tools::shared_ptr<viennacl::hsa::kernel>(new kernel(hsa_code_descriptor, buffer,*this,*p_context_, handle_.get().kernels_[i].name_ )));
	    }
	    return status;
  }


private:
 static hsa_status_t get_kernarg(hsa_region_t region, void* data) {
    hsa_region_flag_t flags;
    hsa_region_get_info(region, HSA_REGION_INFO_FLAGS, &flags);
    if (flags & HSA_REGION_FLAG_KERNARG) {
      hsa_region_t * ret = (hsa_region_t *) data;
      *ret = region;
      return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
  }

private:

  viennacl::hsa::handle<brig_module> handle_;
  viennacl::hsa::context const * p_context_;
  std::string name_;
  kernel_container_type kernels_;
};

} //namespace hsa
} //namespace viennacl


#endif
