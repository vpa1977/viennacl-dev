#ifndef VIENNACL_LINALG_HSA_ILU_OPERATIONS_HPP_
#define VIENNACL_LINALG_HSA_ILU_OPERATIONS_HPP_

/* =========================================================================
   Copyright (c) 2010-2015, Institute for Microelectronics,
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

/** @file viennacl/linalg/opencl/ilu_operations.hpp
    @brief Implementations of specialized routines for the Chow-Patel parallel ILU preconditioner using OpenCL
*/

#include <cmath>
#include <algorithm>  //for std::max and std::min

#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/hsa/common.hpp"
#include "viennacl/linalg/opencl/kernels/ilu.hpp"
#include "viennacl/meta/predicate.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/traits/stride.hpp"
#include "viennacl/linalg/vector_operations.hpp"


namespace viennacl
{
namespace linalg
{
namespace hsa
{

/////////////////////// ICC /////////////////////

template<typename NumericT>
void extract_L(compressed_matrix<NumericT> const & A,
                compressed_matrix<NumericT>       & L)
{
  viennacl::hsa::context & ctx = const_cast<viennacl::hsa::context &>(viennacl::traits::hsa_context(A));
  viennacl::linalg::opencl::kernels::ilu<NumericT, viennacl::hsa::context>::init(ctx);

  //
  // Step 1: Count elements in L:
  //
  viennacl::hsa::kernel & k1 = ctx.get_kernel(viennacl::linalg::opencl::kernels::ilu<NumericT>::program_name(), "extract_L_1");

  viennacl::hsa::enqueue(k1(A.handle1().hsa_handle(), A.handle2().hsa_handle(), cl_uint(A.size1()),
                            L.handle1().hsa_handle())
                        );

  //
  // Step 2: Exclusive scan on row_buffers:
  //
  viennacl::vector_base<unsigned int> wrapped_L_row_buffer(L.handle1(), A.size1() + 1, 0, 1);
  viennacl::linalg::exclusive_scan(wrapped_L_row_buffer, wrapped_L_row_buffer);
  L.reserve(wrapped_L_row_buffer[L.size1()], false);


  //
  // Step 3: Write entries
  //
  viennacl::hsa::kernel & k2 = ctx.get_kernel(viennacl::linalg::opencl::kernels::ilu<NumericT>::program_name(), "extract_L_2");

  viennacl::hsa::enqueue(k2(A.handle1().hsa_handle(), A.handle2().hsa_handle(), A.handle().hsa_handle(), cl_uint(A.size1()),
                            L.handle1().hsa_handle(), L.handle2().hsa_handle(), L.handle().hsa_handle())
                        );

  L.generate_row_block_information();

} // extract_LU

///////////////////////////////////////////////



/** @brief Scales the values extracted from A such that A' = DAD has unit diagonal. Updates values from A in L and U accordingly. */
template<typename NumericT>
void icc_scale(compressed_matrix<NumericT> const & A,
               compressed_matrix<NumericT>       & L)
{
  viennacl::vector<NumericT> D(A.size1(), viennacl::traits::context(A));

  viennacl::hsa::context & ctx = const_cast<viennacl::hsa::context &>(viennacl::traits::hsa_context(A));
  viennacl::linalg::opencl::kernels::ilu<NumericT, viennacl::hsa::context>::init(ctx);

  // fill D:
  viennacl::hsa::kernel & k1 = ctx.get_kernel(viennacl::linalg::opencl::kernels::ilu<NumericT>::program_name(), "ilu_scale_kernel_1");
  viennacl::hsa::enqueue(k1(A.handle1().hsa_handle(), A.handle2().hsa_handle(), A.handle().hsa_handle(), cl_uint(A.size1()), D) );

  // scale L:
  viennacl::hsa::kernel & k2 = ctx.get_kernel(viennacl::linalg::opencl::kernels::ilu<NumericT>::program_name(), "ilu_scale_kernel_2");
  viennacl::hsa::enqueue(k2(L.handle1().hsa_handle(), L.handle2().hsa_handle(), L.handle().hsa_handle(), cl_uint(A.size1()), D) );

}

/////////////////////////////////////


/** @brief Performs one nonlinear relaxation step in the Chow-Patel-ILU using OpenMP (cf. Algorithm 2 in paper) */
template<typename NumericT>
void icc_chow_patel_sweep(compressed_matrix<NumericT>       & L,
                          vector<NumericT>            const & aij_L)
{
  viennacl::hsa::context & ctx = const_cast<viennacl::hsa::context &>(viennacl::traits::hsa_context(L));
  viennacl::linalg::opencl::kernels::ilu<NumericT, viennacl::hsa::context>::init(ctx);

  viennacl::backend::mem_handle L_backup;
  viennacl::backend::memory_create(L_backup, L.handle().raw_size(), viennacl::traits::context(L));
  viennacl::backend::memory_copy(L.handle(), L_backup, 0, 0, L.handle().raw_size());

  viennacl::hsa::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::ilu<NumericT>::program_name(), "icc_chow_patel_sweep_kernel");
  viennacl::hsa::enqueue(k(L.handle1().hsa_handle(), L.handle2().hsa_handle(), L.handle().hsa_handle(), L_backup.hsa_handle(), cl_uint(L.size1()),
                           aij_L)
                        );

}


/////////////////////// ILU /////////////////////

template<typename NumericT>
void extract_LU(compressed_matrix<NumericT> const & A,
                compressed_matrix<NumericT>       & L,
                compressed_matrix<NumericT>       & U)
{
  viennacl::hsa::context & ctx = const_cast<viennacl::hsa::context &>(viennacl::traits::hsa_context(A));
  viennacl::linalg::opencl::kernels::ilu<NumericT, viennacl::hsa::context>::init(ctx);

  //
  // Step 1: Count elements in L and U:
  //
  viennacl::hsa::kernel & k1 = ctx.get_kernel(viennacl::linalg::opencl::kernels::ilu<NumericT>::program_name(), "extract_LU_1");

  viennacl::hsa::enqueue(k1(A.handle1().hsa_handle(), A.handle2().hsa_handle(), cl_uint(A.size1()),
                            L.handle1().hsa_handle(),
                            U.handle1().hsa_handle())
                        );

  //
  // Step 2: Exclusive scan on row_buffers:
  //
  viennacl::vector_base<unsigned int> wrapped_L_row_buffer(L.handle1(), A.size1() + 1, 0, 1);
  viennacl::linalg::exclusive_scan(wrapped_L_row_buffer, wrapped_L_row_buffer);
  L.reserve(wrapped_L_row_buffer[L.size1()], false);

  viennacl::vector_base<unsigned int> wrapped_U_row_buffer(U.handle1(), A.size1() + 1, 0, 1);
  viennacl::linalg::exclusive_scan(wrapped_U_row_buffer, wrapped_U_row_buffer);
  U.reserve(wrapped_U_row_buffer[U.size1()], false);

  //
  // Step 3: Write entries
  //
  viennacl::hsa::kernel & k2 = ctx.get_kernel(viennacl::linalg::opencl::kernels::ilu<NumericT>::program_name(), "extract_LU_2");

  viennacl::hsa::enqueue(k2(A.handle1().hsa_handle(), A.handle2().hsa_handle(), A.handle().hsa_handle(), cl_uint(A.size1()),
                            L.handle1().hsa_handle(), L.handle2().hsa_handle(), L.handle().hsa_handle(),
                            U.handle1().hsa_handle(), U.handle2().hsa_handle(), U.handle().hsa_handle())
                        );

  L.generate_row_block_information();
  // Note: block information for U will be generated after transposition

} // extract_LU

///////////////////////////////////////////////



/** @brief Scales the values extracted from A such that A' = DAD has unit diagonal. Updates values from A in L and U accordingly. */
template<typename NumericT>
void ilu_scale(compressed_matrix<NumericT> const & A,
               compressed_matrix<NumericT>       & L,
               compressed_matrix<NumericT>       & U)
{
  viennacl::vector<NumericT> D(A.size1(), viennacl::traits::context(A));

  viennacl::hsa::context & ctx = const_cast<viennacl::hsa::context &>(viennacl::traits::hsa_context(A));
  viennacl::linalg::opencl::kernels::ilu<NumericT, viennacl::hsa::context>::init(ctx);

  // fill D:
  viennacl::hsa::kernel & k1 = ctx.get_kernel(viennacl::linalg::opencl::kernels::ilu<NumericT>::program_name(), "ilu_scale_kernel_1");
  viennacl::hsa::enqueue(k1(A.handle1().hsa_handle(), A.handle2().hsa_handle(), A.handle().hsa_handle(), cl_uint(A.size1()), D) );

  // scale L:
  viennacl::hsa::kernel & k2 = ctx.get_kernel(viennacl::linalg::opencl::kernels::ilu<NumericT>::program_name(), "ilu_scale_kernel_2");
  viennacl::hsa::enqueue(k2(L.handle1().hsa_handle(), L.handle2().hsa_handle(), L.handle().hsa_handle(), cl_uint(A.size1()), D) );

  // scale U:
  viennacl::hsa::enqueue(k2(U.handle1().hsa_handle(), U.handle2().hsa_handle(), U.handle().hsa_handle(), cl_uint(A.size1()), D) );

}

/////////////////////////////////////


/** @brief Performs one nonlinear relaxation step in the Chow-Patel-ILU using OpenMP (cf. Algorithm 2 in paper) */
template<typename NumericT>
void ilu_chow_patel_sweep(compressed_matrix<NumericT>       & L,
                          vector<NumericT>            const & aij_L,
                          compressed_matrix<NumericT>       & U_trans,
                          vector<NumericT>            const & aij_U_trans)
{
  viennacl::hsa::context & ctx = const_cast<viennacl::hsa::context &>(viennacl::traits::hsa_context(L));
  viennacl::linalg::opencl::kernels::ilu<NumericT, viennacl::hsa::context>::init(ctx);

  viennacl::backend::mem_handle L_backup;
  viennacl::backend::memory_create(L_backup, L.handle().raw_size(), viennacl::traits::context(L));
  viennacl::backend::memory_copy(L.handle(), L_backup, 0, 0, L.handle().raw_size());

  viennacl::backend::mem_handle U_backup;
  viennacl::backend::memory_create(U_backup, U_trans.handle().raw_size(), viennacl::traits::context(U_trans));
  viennacl::backend::memory_copy(U_trans.handle(), U_backup, 0, 0, U_trans.handle().raw_size());

  viennacl::hsa::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::ilu<NumericT>::program_name(), "ilu_chow_patel_sweep_kernel");
  viennacl::hsa::enqueue(k(L.handle1().hsa_handle(), L.handle2().hsa_handle(), L.handle().hsa_handle(), L_backup.hsa_handle(), cl_uint(L.size1()),
                           aij_L,
                           U_trans.handle1().hsa_handle(), U_trans.handle2().hsa_handle(), U_trans.handle().hsa_handle(), U_backup.hsa_handle(),
                           aij_U_trans)
                        );

}

//////////////////////////////////////



template<typename NumericT>
void ilu_form_neumann_matrix(compressed_matrix<NumericT> & R,
                             vector<NumericT> & diag_R)
{
  viennacl::hsa::context & ctx = const_cast<viennacl::hsa::context &>(viennacl::traits::hsa_context(R));
  viennacl::linalg::opencl::kernels::ilu<NumericT, viennacl::hsa::context>::init(ctx);

  viennacl::hsa::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::ilu<NumericT>::program_name(), "ilu_form_neumann_matrix_kernel");
  viennacl::hsa::enqueue(k(R.handle1().hsa_handle(), R.handle2().hsa_handle(), R.handle().hsa_handle(), cl_uint(R.size1()),
                           diag_R)
                        );
}

} //namespace hsa
} //namespace linalg
} //namespace viennacl


#endif
