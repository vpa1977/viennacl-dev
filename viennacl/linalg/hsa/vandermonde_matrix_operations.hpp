#ifndef VIENNACL_LINALG_HSA_VANDERMONDE_MATRIX_OPERATIONS_HPP_
#define VIENNACL_LINALG_HSA_VANDERMONDE_MATRIX_OPERATIONS_HPP_

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

/** @file viennacl/linalg/opencl/vandermonde_matrix_operations.hpp
    @brief Implementations of operations using vandermonde_matrix
*/

#include "viennacl/forwards.h"
#include "viennacl/hsa/backend.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/fft.hpp"
#include "viennacl/linalg/opencl/kernels/fft.hpp"

namespace viennacl
{
namespace linalg
{
namespace hsa
{

/** @brief Carries out matrix-vector multiplication with a vandermonde_matrix
*
* Implementation of the convenience expression y = prod(A, x);
*
* @param A    The Vandermonde matrix
* @param x    The vector
* @param y    The result vector
*/
template<typename NumericT, unsigned int AlignmentV>
void prod_impl(viennacl::vandermonde_matrix<NumericT, AlignmentV> const & A,
               viennacl::vector_base<NumericT> const & x,
               viennacl::vector_base<NumericT>       & y)
{
  viennacl::hsa::context & ctx = const_cast<viennacl::hsa::context &>(viennacl::traits::hsa_context(A));
  viennacl::linalg::opencl::kernels::fft<NumericT, viennacl::hsa::context>::init(ctx);

  viennacl::hsa::kernel & kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::fft<NumericT>::program_name(), "vandermonde_prod");
  viennacl::hsa::enqueue(kernel(viennacl::traits::hsa_handle(A),
                                viennacl::traits::hsa_handle(x),
                                viennacl::traits::hsa_handle(y),
                                static_cast<cl_uint>(A.size1())));
}

} //namespace hsa
} //namespace linalg
} //namespace viennacl


#endif
