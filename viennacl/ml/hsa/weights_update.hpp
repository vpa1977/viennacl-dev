/*
 * weights_update.hpp
 *
 *  Created on: 7/05/2015
 *      Author: bsp
 */

#ifndef VIENNACL_ML_HSA_WEIGHTS_UPDATE_HPP_
#define VIENNACL_ML_HSA_WEIGHTS_UPDATE_HPP_

#include "viennacl/forwards.h"
#include "viennacl/hsa/device.hpp"
#include "viennacl/hsa/handle.hpp"
#include "viennacl/hsa/kernel.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/hsa/common.hpp"
#include "viennacl/ml/hsa/ml_kernels.hpp"


namespace viennacl
{
namespace ml
{
namespace hsa
{

	template <typename T>
	void sgd_update_weights(viennacl::vector_base<T>& weights, const viennacl::compressed_matrix<T>& batch, const viennacl::vector_base<T>& factors)
	{
		viennacl::hsa::context & ctx = const_cast<viennacl::hsa::context &>(viennacl::traits::hsa_handle(weights).context());
		sgd_kernels<double>::init(ctx);
		static viennacl::hsa::kernel& update_by_factor_kernel = ctx.get_kernel(sgd_kernels<T>::program_name(), "sgd_update_weights");
		//double* elements,double * factors, int* rows, int * columns)
		viennacl::hsa::enqueue(update_by_factor_kernel(batch.handle().hsa_handle(), factors.handle().hsa_handle(), batch.handle1().hsa_handle(), batch.handle2().hsa_handle()));

		viennacl::vector<T> intermediate_sum(weights.size());
		T   const * elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<T>(batch.handle());
		unsigned int const * row_buffer =viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(batch.handle1());
		unsigned int const * col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(batch.handle2());
		for (int i = 0 ; i < batch.size1(); ++i)
		{
			int start = row_buffer[i];
			int end = row_buffer[i+1];
			for (int column = start; column < end ; ++column)
			{
				intermediate_sum( col_buffer[column] ) += elements[column];
			}
		}
		weights += intermediate_sum;

	}
}

}
}



#endif /* VIENNACL_ML_WEIGHTS_UPDATE_HPP_ */
