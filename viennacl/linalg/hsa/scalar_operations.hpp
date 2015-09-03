#ifndef VIENNACL_LINALG_HSA_SCALAR_OPERATIONS_HPP_
#define VIENNACL_LINALG_HSA_SCALAR_OPERATIONS_HPP_

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

/** @file viennacl/linalg/opencl/scalar_operations.hpp
    @brief Implementations of scalar operations using OpenCL
*/

#include "viennacl/forwards.h"
#include "viennacl/hsa/device.hpp"
#include "viennacl/hsa/handle.hpp"
#include "viennacl/hsa/kernel.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/opencl/kernels/scalar.hpp"
#include "viennacl/linalg/hsa/common.hpp"
#include "viennacl/meta/predicate.hpp"
#include "viennacl/meta/result_of.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/traits/stride.hpp"
#include "viennacl/traits/handle.hpp"

namespace viennacl
{
namespace linalg
{
namespace hsa
{

template<typename ScalarT1,
         typename ScalarT2, typename NumericT>
typename viennacl::enable_if< viennacl::is_scalar<ScalarT1>::value
                              && viennacl::is_scalar<ScalarT2>::value
                              && viennacl::is_any_scalar<NumericT>::value
                            >::type
as(ScalarT1 & s1,
   ScalarT2 const & s2, NumericT const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha)
{
//assert_context()

  typedef typename viennacl::result_of::cpu_value_type<ScalarT1>::type        value_type;
  viennacl::hsa::context & ctx = const_cast<viennacl::hsa::context &>(viennacl::traits::hsa_context(s1));
  viennacl::linalg::opencl::kernels::scalar<value_type, viennacl::hsa::context>::init(ctx);

  cl_uint options_alpha = detail::make_options(len_alpha, reciprocal_alpha, flip_sign_alpha);

  bool is_cpu = viennacl::is_cpu_scalar<NumericT>::value;
  viennacl::hsa::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::scalar<value_type>::program_name(), is_cpu ? "as_cpu" : "as_gpu");
  k.local_work_size(0, 1);
  k.global_work_size(0, 1);
  viennacl::hsa::enqueue(k(viennacl::traits::hsa_handle(s1),
                           viennacl::traits::hsa_handle(viennacl::tools::promote_if_host_scalar<value_type>(alpha)),
                           options_alpha,
                           viennacl::traits::hsa_handle(s2) )
                        );
}


template<typename ScalarT1,
         typename ScalarT2, typename NumericT2,
         typename ScalarT3, typename NumericT3>
typename viennacl::enable_if< viennacl::is_scalar<ScalarT1>::value
                              && viennacl::is_scalar<ScalarT2>::value
                              && viennacl::is_scalar<ScalarT3>::value
                              && viennacl::is_any_scalar<NumericT2>::value
                              && viennacl::is_any_scalar<NumericT3>::value
                            >::type
asbs(ScalarT1 & s1,
     ScalarT2 const & s2, NumericT2 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
     ScalarT3 const & s3, NumericT3 const & beta,  vcl_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
{
//assert_context()
//assert_context()

  typedef typename viennacl::result_of::cpu_value_type<ScalarT1>::type        value_type;
  viennacl::hsa::context & ctx = const_cast<viennacl::hsa::context &>(viennacl::traits::hsa_context(s1));
  viennacl::linalg::opencl::kernels::scalar<value_type, viennacl::hsa::context>::init(ctx);

  std::string kernel_name;
  bool is_cpu_2 = viennacl::is_cpu_scalar<NumericT2>::value;
  bool is_cpu_3 = viennacl::is_cpu_scalar<NumericT3>::value;
  if (is_cpu_2 && is_cpu_3)
    kernel_name = "asbs_cpu_cpu";
  else if (is_cpu_2 && !is_cpu_3)
    kernel_name = "asbs_cpu_gpu";
  else if (!is_cpu_2 && is_cpu_3)
    kernel_name = "asbs_gpu_cpu";
  else
    kernel_name = "asbs_gpu_gpu";

  cl_uint options_alpha = detail::make_options(len_alpha, reciprocal_alpha, flip_sign_alpha);
  cl_uint options_beta  = detail::make_options(len_beta,  reciprocal_beta,  flip_sign_beta);

  viennacl::hsa::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::scalar<value_type>::program_name(), kernel_name);
  k.local_work_size(0, 1);
  k.global_work_size(0, 1);
  viennacl::hsa::enqueue(k(viennacl::traits::hsa_handle(s1),
                           viennacl::traits::hsa_handle(viennacl::tools::promote_if_host_scalar<value_type>(alpha)),
                           options_alpha,
                           viennacl::traits::hsa_handle(s2),
                           viennacl::traits::hsa_handle(viennacl::tools::promote_if_host_scalar<value_type>(beta)),
                           options_beta,
                           viennacl::traits::hsa_handle(s3) )
                        );
}


template<typename ScalarT1,
         typename ScalarT2, typename NumericT2,
         typename ScalarT3, typename NumericT3>
typename viennacl::enable_if< viennacl::is_scalar<ScalarT1>::value
                              && viennacl::is_scalar<ScalarT2>::value
                              && viennacl::is_scalar<ScalarT3>::value
                              && viennacl::is_any_scalar<NumericT2>::value
                              && viennacl::is_any_scalar<NumericT3>::value
                            >::type
asbs_s(ScalarT1 & s1,
       ScalarT2 const & s2, NumericT2 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
       ScalarT3 const & s3, NumericT3 const & beta,  vcl_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
{
//assert_context()
//assert_context()

  typedef typename viennacl::result_of::cpu_value_type<ScalarT1>::type        value_type;
  viennacl::hsa::context & ctx = const_cast<viennacl::hsa::context &>(viennacl::traits::hsa_context(s1));
  viennacl::linalg::opencl::kernels::scalar<value_type, viennacl::hsa::context>::init(ctx);

  std::string kernel_name;
  if (viennacl::is_cpu_scalar<NumericT2>::value && viennacl::is_cpu_scalar<NumericT3>::value)
    kernel_name = "asbs_s_cpu_cpu";
  else if (viennacl::is_cpu_scalar<NumericT2>::value && !viennacl::is_cpu_scalar<NumericT3>::value)
    kernel_name = "asbs_s_cpu_gpu";
  else if (!viennacl::is_cpu_scalar<NumericT2>::value && viennacl::is_cpu_scalar<NumericT3>::value)
    kernel_name = "asbs_s_gpu_cpu";
  else
    kernel_name = "asbs_s_gpu_gpu";

  cl_uint options_alpha = detail::make_options(len_alpha, reciprocal_alpha, flip_sign_alpha);
  cl_uint options_beta  = detail::make_options(len_beta,  reciprocal_beta,  flip_sign_beta);

  viennacl::hsa::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::scalar<value_type>::program_name(), kernel_name);
  k.local_work_size(0, 1);
  k.global_work_size(0, 1);
  viennacl::hsa::enqueue(k(viennacl::traits::hsa_handle(s1),
                           viennacl::traits::hsa_handle(viennacl::tools::promote_if_host_scalar<value_type>(alpha)),
                           options_alpha,
                           viennacl::traits::hsa_handle(s2),
                           viennacl::traits::hsa_handle(viennacl::tools::promote_if_host_scalar<value_type>(beta)),
                           options_beta,
                           viennacl::traits::hsa_handle(s3) )
                        );
}


/** @brief Swaps the contents of two scalars, data is copied
*
* @param s1   The first scalar
* @param s2   The second scalar
*/
template<typename ScalarT1, typename ScalarT2>
typename viennacl::enable_if<    viennacl::is_scalar<ScalarT1>::value
                              && viennacl::is_scalar<ScalarT2>::value
                            >::type
swap(ScalarT1 & s1, ScalarT2 & s2)
{
//assert_context()

  typedef typename viennacl::result_of::cpu_value_type<ScalarT1>::type        value_type;
  viennacl::hsa::context & ctx = const_cast<viennacl::hsa::context &>(viennacl::traits::hsa_context(s1));
  viennacl::linalg::opencl::kernels::scalar<value_type, viennacl::hsa::context>::init(ctx);

  viennacl::hsa::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::scalar<value_type>::program_name(), "swap");
  k.local_work_size(0, 1);
  k.global_work_size(0, 1);
  viennacl::hsa::enqueue(k(viennacl::traits::hsa_handle(s1),
                           viennacl::traits::hsa_handle(s2))
                        );
}



} //namespace hsa
} //namespace linalg
} //namespace viennacl


#endif
