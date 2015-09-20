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
		virtual ~compatible_handle() {};

	};

}
#endif	/* COMPATIBLE_HANDLE_HPP */

