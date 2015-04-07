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

namespace viennacl {
namespace hsa {

/** @brief Wrapper class for an OpenCL program.
 *
 * This class was written when the OpenCL C++ bindings haven't been standardized yet.
 * Regardless, it takes care about some additional details and is supposed to provide higher convenience by holding the kernels defined in the program.
 */
class program {
	typedef std::vector<tools::shared_ptr<viennacl::hsa::kernel> > kernel_container_type;

public:
	program() :
			p_context_(NULL) {
	}
	program(const std::vector<char>& program_module,
			viennacl::hsa::context const & program_context,
			std::string const & prog_name = std::string()) :
			source_(program_module), p_context_(&program_context), name_(
					prog_name) {
		finalize();
	}

	program(program const & other) :
			source_(other.source_), p_context_(other.p_context_), name_(
					other.name_), kernels_(other.kernels_) {
		finalize();
	}

	viennacl::hsa::program & operator=(const program & other) {
		source_ = other.source_;
		name_ = other.name_;
		p_context_ = other.p_context_;
		finalize();
		return *this;
	}

	viennacl::hsa::context const * p_context() const {
		return p_context_;
	}

	std::string const & name() const {
		return name_;
	}

	/** @brief Returns the kernel with the provided name */
	inline viennacl::hsa::kernel & get_kernel(std::string const & name); //see context.hpp for implementation

private:
	friend class viennacl::hsa::context;

	void add_kernel(
			const tools::shared_ptr<viennacl::hsa::kernel>& new_kernel) {
		kernels_.push_back(new_kernel);
	}

	// extract kernels and launch parameters
	void finalize();
	void load_kernels();

	std::vector<char> source_;
	handle<hsa_code> handle_;
	viennacl::hsa::context const * p_context_;
	std::string name_;
	kernel_container_type kernels_;
	hsa_region_t kernarg_region_;
};

} //namespace hsa
} //namespace viennacl

#endif
