#ifndef VIENNACL_LINALG_HSA_MISC_OPERATIONS_HPP_
#define VIENNACL_LINALG_HSA_MISC_OPERATIONS_HPP_

/* =========================================================================
   Copyright (c) 2010-2015, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/** @file viennacl/linalg/opencl/misc_operations.hpp
    @brief Implementations of operations using compressed_matrix and OpenCL
*/

#include "viennacl/forwards.h"
#include "viennacl/hsa/device.hpp"
#include "viennacl/hsa/handle.hpp"
#include "viennacl/hsa/kernel.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/opencl/kernels/ilu.hpp"


namespace viennacl
{
namespace linalg
{
namespace hsa
{
namespace detail
{

template<typename NumericT>
void level_scheduling_substitute(vector<NumericT> & x,
                                 viennacl::backend::mem_handle const & row_index_array,
                                 viennacl::backend::mem_handle const & row_buffer,
                                 viennacl::backend::mem_handle const & col_buffer,
                                 viennacl::backend::mem_handle const & element_buffer,
                                 vcl_size_t num_rows
                                )
{
  viennacl::hsa::context & ctx = const_cast<viennacl::hsa::context &>(viennacl::traits::hsa_context(x));

  viennacl::linalg::opencl::kernels::ilu<NumericT, viennacl::hsa::context>::init(ctx);
  viennacl::hsa::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::ilu<NumericT>::program_name(), "level_scheduling_substitute");

  viennacl::hsa::enqueue(k(row_index_array.hsa_handle(), row_buffer.hsa_handle(), col_buffer.hsa_handle(), element_buffer.hsa_handle(),
                           x,
                           static_cast<cl_uint>(num_rows)));
}

} //namespace detail
} // namespace hsa
} //namespace linalg
} //namespace viennacl


#endif
