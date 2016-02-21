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

/** @file viennacl/linalg/opencl/fft_operations.hpp
 @brief Implementations of Fast Furier Transformation using OpenCL
 */
#ifndef VIENNACL_LINALG_HSA_FFT_OPERATIONS_HPP_

#define VIENNACL_LINALG_HSA_FFT_OPERATIONS_HPP_

#include "viennacl/forwards.h"
#include "viennacl/hsa/device.hpp"
#include "viennacl/hsa/kernel.hpp"
#include "viennacl/hsa/handle.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/traits/stride.hpp"
#include "viennacl/linalg/host_based/fft_operations.hpp"
#include "viennacl/linalg/opencl/kernels/fft.hpp"
#include "viennacl/linalg/opencl/kernels/matrix.hpp"

#include <viennacl/vector.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/traits/handle.hpp>

#include <cmath>
#include <stdexcept>

namespace viennacl
{
namespace linalg
{

namespace hsa
{

/**
 * @brief Direct algorithm for computing Fourier transformation.
 *
 * Works on any sizes of data.
 * Serial implementation has o(n^2) complexity
 */
template<typename NumericT, unsigned int AlignmentV>
void direct(viennacl::vector<NumericT, AlignmentV> const & in,
            viennacl::vector<NumericT, AlignmentV> const & out,
            vcl_size_t size, vcl_size_t stride, vcl_size_t batch_num, NumericT sign = NumericT(-1),
            viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order = viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
{
  viennacl::hsa::context & ctx = const_cast<viennacl::hsa::context &>(viennacl::traits::context(in).hsa_context());
  viennacl::linalg::opencl::kernels::fft<NumericT, viennacl::hsa::context>::init(ctx);

  std::string program_string = viennacl::linalg::opencl::kernels::matrix_legacy<NumericT, row_major>::program_name();
  if (data_order == viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::COL_MAJOR)
  {
    viennacl::linalg::opencl::kernels::matrix_legacy<NumericT, column_major, viennacl::hsa::context>::init(ctx);
    program_string =
        viennacl::linalg::opencl::kernels::matrix_legacy<NumericT, column_major>::program_name();
  } else
    viennacl::linalg::opencl::kernels::matrix_legacy<NumericT, row_major, viennacl::hsa::context>::init(ctx);

  viennacl::hsa::kernel & k = ctx.get_kernel(program_string, "fft_direct");
  viennacl::hsa::enqueue(k(in, out,
                           static_cast<cl_uint>(size),
                           static_cast<cl_uint>(stride),
                           static_cast<cl_uint>(batch_num),
                           sign)
                        );
}

/*
 * This function performs reorder of input data. Indexes are sorted in bit-reversal order.
 * Such reordering should be done before in-place FFT.
 */
template<typename NumericT, unsigned int AlignmentV>
void reorder(viennacl::vector<NumericT, AlignmentV> const & in,
             vcl_size_t size, vcl_size_t stride,
             vcl_size_t bits_datasize, vcl_size_t batch_num,
             viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order = viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
{
  viennacl::hsa::context & ctx = const_cast<viennacl::hsa::context &>(viennacl::traits::context(in).hsa_context());
  viennacl::linalg::opencl::kernels::fft<NumericT, viennacl::hsa::context>::init(ctx);

  std::string program_string = viennacl::linalg::opencl::kernels::matrix_legacy<NumericT, row_major>::program_name();
  if (data_order == viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::COL_MAJOR)
  {
    viennacl::linalg::opencl::kernels::matrix_legacy<NumericT, column_major, viennacl::hsa::context>::init(ctx);
    program_string = viennacl::linalg::opencl::kernels::matrix_legacy<NumericT, column_major>::program_name();
  } else
    viennacl::linalg::opencl::kernels::matrix_legacy<NumericT, row_major, viennacl::hsa::context>::init(ctx);

  viennacl::hsa::kernel& k = ctx.get_kernel(program_string, "fft_reorder");
  viennacl::hsa::enqueue(k(in,
                           static_cast<cl_uint>(bits_datasize), static_cast<cl_uint>(size),
                           static_cast<cl_uint>(stride), static_cast<cl_uint>(batch_num))
                        );
}

/**
 * @brief Radix-2 algorithm for computing Fourier transformation.
 *
 * Works only on power-of-two sizes of data.
 * Serial implementation has o(n * lg n) complexity.
 * This is a Cooley-Tukey algorithm
 */
template<typename NumericT, unsigned int AlignmentV>
void radix2(viennacl::vector<NumericT, AlignmentV> const & in,
            vcl_size_t size, vcl_size_t stride,
            vcl_size_t batch_num, NumericT sign = NumericT(-1),
            viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order = viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
{
    viennacl::hsa::context & ctx = const_cast<viennacl::hsa::context &>(viennacl::traits::context(in).hsa_context());
  viennacl::linalg::opencl::kernels::fft<NumericT, viennacl::hsa::context>::init(ctx);

  assert(batch_num != 0 && bool("batch_num must be larger than 0"));

  std::string program_string = viennacl::linalg::opencl::kernels::matrix_legacy<NumericT, row_major>::program_name();
  if (data_order == viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::COL_MAJOR)
  {
    viennacl::linalg::opencl::kernels::matrix_legacy<NumericT, column_major, viennacl::hsa::context>::init(ctx);
    program_string = viennacl::linalg::opencl::kernels::matrix_legacy<NumericT, column_major>::program_name();
  } else
    viennacl::linalg::opencl::kernels::matrix_legacy<NumericT, row_major, viennacl::hsa::context>::init(ctx);

  vcl_size_t bits_datasize = viennacl::linalg::detail::fft::num_bits(size);
  if (size <= viennacl::linalg::detail::fft::MAX_LOCAL_POINTS_NUM)
  {
    viennacl::hsa::kernel & k = ctx.get_kernel(program_string, "fft_radix2_local");
    viennacl::hsa::enqueue(k(in,
                             viennacl::hsa::local_mem((size * 4) * sizeof(NumericT)),
                             static_cast<cl_uint>(bits_datasize), static_cast<cl_uint>(size),
                             static_cast<cl_uint>(stride), static_cast<cl_uint>(batch_num), sign));

  }
  else
  {
	 viennacl::hsa::kernel & k = ctx.get_kernel(program_string, "fft_radix2");
     viennacl::linalg::hsa::reorder<NumericT>(in, size, stride, bits_datasize, batch_num);
    for (vcl_size_t step = 0; step < bits_datasize; step++)
    {

      viennacl::hsa::enqueue(k(in,
                               static_cast<cl_uint>(step), static_cast<cl_uint>(bits_datasize),
                               static_cast<cl_uint>(size), static_cast<cl_uint>(stride),
                               static_cast<cl_uint>(batch_num), sign));
    }
  }
}

/**
 * @brief Bluestein's algorithm for computing Fourier transformation.
 *
 * Currently,  Works only for sizes of input data which less than 2^16.
 * Uses a lot of additional memory, but should be fast for any size of data.
 * Serial implementation has something about o(n * lg n) complexity
 */
template<typename NumericT, unsigned int AlignmentV>
void bluestein(viennacl::vector<NumericT, AlignmentV>& in,
               viennacl::vector<NumericT, AlignmentV>& out, vcl_size_t /*batch_num*/)
{
  viennacl::hsa::context & ctx = const_cast<viennacl::hsa::context &>(viennacl::traits::hsa_context(in));
  viennacl::linalg::opencl::kernels::fft<NumericT, viennacl::hsa::context>::init(ctx);

  vcl_size_t size = in.size() >> 1;
  vcl_size_t ext_size = viennacl::linalg::detail::fft::next_power_2(2 * size - 1);

  viennacl::vector<NumericT, AlignmentV> A(ext_size << 1);
  viennacl::vector<NumericT, AlignmentV> B(ext_size << 1);
  viennacl::vector<NumericT, AlignmentV> Z(ext_size << 1);

  {
    viennacl::hsa::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::fft<NumericT>::program_name(), "zero2");
    viennacl::hsa::enqueue(k(A, B, static_cast<cl_uint>(ext_size)));
  }
  {
    viennacl::hsa::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::fft<NumericT>::program_name(), "bluestein_pre");
    viennacl::hsa::enqueue(k(in, A, B, static_cast<cl_uint>(size), static_cast<cl_uint>(ext_size)));
  }

  viennacl::linalg::convolve_i(A, B, Z);

  {
    viennacl::hsa::kernel& k = ctx.get_kernel(viennacl::linalg::opencl::kernels::fft<NumericT>::program_name(), "bluestein_post");
    viennacl::hsa::enqueue(k(Z, out, static_cast<cl_uint>(size)));
  }
}

/**
 * @brief Mutiply two complex vectors and store result in output
 */
template<typename NumericT, unsigned int AlignmentV>
void multiply_complex(viennacl::vector<NumericT, AlignmentV> const & input1,
                      viennacl::vector<NumericT, AlignmentV> const & input2,
                      viennacl::vector<NumericT, AlignmentV>       & output)
{
  viennacl::hsa::context & ctx = const_cast<viennacl::hsa::context &>(viennacl::traits::hsa_context(input1));
  viennacl::linalg::opencl::kernels::fft<NumericT, viennacl::hsa::context>::init(ctx);
  vcl_size_t size = input1.size() >> 1;
  viennacl::hsa::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::fft<NumericT>::program_name(), "fft_mult_vec");
  viennacl::hsa::enqueue(k(input1, input2, output, static_cast<cl_uint>(size)));
}

/**
 * @brief Normalize vector on with his own size
 */
template<typename NumericT, unsigned int AlignmentV>
void normalize(viennacl::vector<NumericT, AlignmentV> & input)
{
  viennacl::hsa::context & ctx = const_cast<viennacl::hsa::context &>(viennacl::traits::hsa_context(input));
  viennacl::linalg::opencl::kernels::fft<NumericT, viennacl::hsa::context>::init(ctx);

  viennacl::hsa::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::fft<NumericT>::program_name(), "fft_div_vec_scalar");

  vcl_size_t size = input.size() >> 1;
  NumericT norm_factor = static_cast<NumericT>(size);
  viennacl::hsa::enqueue(k(input, static_cast<cl_uint>(size), norm_factor));
}

/**
 * @brief Inplace_transpose matrix
 */
template<typename NumericT, unsigned int AlignmentV>
void transpose(viennacl::matrix<NumericT, viennacl::row_major, AlignmentV> & input)
{
  viennacl::hsa::context & ctx = const_cast<viennacl::hsa::context &>(viennacl::traits::hsa_context(input));
  viennacl::linalg::opencl::kernels::fft<NumericT, viennacl::hsa::context>::init(ctx);

  viennacl::hsa::kernel& k = ctx.get_kernel(viennacl::linalg::opencl::kernels::fft<NumericT>::program_name(), "transpose_inplace");
  viennacl::hsa::enqueue(k(input, static_cast<cl_uint>(input.internal_size1() >> 1),
                           static_cast<cl_uint>(input.internal_size2()) >> 1));
}

/**
 * @brief Transpose matrix
 */
template<typename NumericT, unsigned int AlignmentV>
void transpose(viennacl::matrix<NumericT, viennacl::row_major, AlignmentV> const & input,
               viennacl::matrix<NumericT, viennacl::row_major, AlignmentV> & output)
{
  viennacl::hsa::context & ctx = const_cast<viennacl::hsa::context &>(viennacl::traits::hsa_context(input));
  viennacl::linalg::opencl::kernels::fft<NumericT, viennacl::hsa::context>::init(ctx);

  viennacl::hsa::kernel& k = ctx.get_kernel(viennacl::linalg::opencl::kernels::fft<NumericT>::program_name(), "transpose");
  viennacl::hsa::enqueue(k(input, output, static_cast<cl_uint>(input.internal_size1() >> 1),
                           static_cast<cl_uint>(input.internal_size2() >> 1)));
}

/**
 * @brief Create complex vector from real vector (even elements(2*k) = real part, odd elements(2*k+1) = imaginary part)
 */
template<typename NumericT>
void real_to_complex(viennacl::vector_base<NumericT> const & in,
                     viennacl::vector_base<NumericT>       & out, vcl_size_t size)
{
  viennacl::hsa::context & ctx = const_cast<viennacl::hsa::context &>(viennacl::traits::hsa_context(in));
  viennacl::linalg::opencl::kernels::fft<NumericT, viennacl::hsa::context>::init(ctx);

  viennacl::hsa::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::fft<NumericT>::program_name(), "real_to_complex");
  viennacl::hsa::enqueue(k(in, out, static_cast<cl_uint>(size)));
}

/**
 * @brief Create real vector from complex vector (even elements(2*k) = real part, odd elements(2*k+1) = imaginary part)
 */
template<typename NumericT>
void complex_to_real(viennacl::vector_base<NumericT> const & in,
                     viennacl::vector_base<NumericT>       & out, vcl_size_t size)
{
  viennacl::hsa::context & ctx = const_cast<viennacl::hsa::context &>(viennacl::traits::hsa_context(in));
  viennacl::linalg::opencl::kernels::fft<NumericT, viennacl::hsa::context>::init(ctx);

  viennacl::hsa::kernel& k = ctx.get_kernel(viennacl::linalg::opencl::kernels::fft<NumericT>::program_name(), "complex_to_real");
  viennacl::hsa::enqueue(k(in, out, static_cast<cl_uint>(size)));
}

/**
 * @brief Reverse vector to oposite order and save it in input vector
 */
template<typename NumericT>
void reverse(viennacl::vector_base<NumericT>& in)
{
  viennacl::hsa::context & ctx = const_cast<viennacl::hsa::context &>(viennacl::traits::hsa_context(in));
  viennacl::linalg::opencl::kernels::fft<NumericT, viennacl::hsa::context>::init(ctx);

  vcl_size_t size = in.size();

  viennacl::hsa::kernel& k = ctx.get_kernel(viennacl::linalg::opencl::kernels::fft<NumericT>::program_name(), "reverse_inplace");
  viennacl::hsa::enqueue(k(in, static_cast<cl_uint>(size)));
}

} //namespace hsa
} //namespace linalg
} //namespace viennacl

#endif /* FFT_OPERATIONS_HPP_ */

