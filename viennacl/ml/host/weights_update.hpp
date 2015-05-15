/*
 * weights_update.hpp
 *
 *  Created on: 7/05/2015
 *      Author: bsp
 */

#ifndef VIENNACL_ML_HOST_WEIGHTS_UPDATE_HPP_
#define VIENNACL_ML_HOST_WEIGHTS_UPDATE_HPP_

#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/vector_proxy.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/meta/predicate.hpp"
#include "viennacl/meta/result_of.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/traits/stride.hpp"
#include "viennacl/linalg/detail/op_applier.hpp"
#include "viennacl/linalg/host_based/common.hpp"


namespace viennacl
{
namespace ml
{
namespace host
{

	template <typename sgd_matrix_type, typename T = double>
	void sgd_update_weights(viennacl::vector_base<T>& weights, const sgd_matrix_type& batch, const viennacl::vector_base<T>& factors)
	{
		/*  T         * result_buf = viennacl::linalg::host_based::detail::extract_raw_pointer<T>(weights);
		  T   const * elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<T>(batch.handle());
		  T   const * factors_buf    = viennacl::linalg::host_based::detail::extract_raw_pointer<T>(factors);
		  unsigned int const * row_buffer =viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(batch.handle1());
		  unsigned int const * col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(batch.handle2());
		#ifdef VIENNACL_WITH_OPENMP
		  #pragma omp parallel for
		#endif
		  for (long row = 0; row < static_cast<long>(batch.size1()); ++row)
		  {
			vcl_size_t row_end = row_buffer[row+1];
			for (vcl_size_t i = row_buffer[row]; i < row_end; ++i)
			{
				result_buf[ col_buffer[i] ] += elements[i] * factors_buf[row];
			}
		  }*/
	}
}

}
}



#endif /* VIENNACL_ML_WEIGHTS_UPDATE_HPP_ */
