/*
 * weights_update.hpp
 *
 *  Created on: 7/05/2015
 *      Author: bsp
 */

#ifndef VIENNACL_ML_OPENCL_WEIGHTS_UPDATE_HPP_
#define VIENNACL_ML_OPENCL_WEIGHTS_UPDATE_HPP_

#include "viennacl/forwards.h"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/handle.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/opencl/common.hpp"
#include "viennacl/ml/opencl/ml_kernels.hpp"


namespace viennacl
{
namespace ml
{
namespace opencl
{
	template <typename T>
	void scan_inclusive( viennacl::vector_base<T>& to_reduce, viennacl::vector_base<T>& reduce_result)
	{
		throw  memory_exception("not implemented");
	}

	template <typename T>
	void sgd_compute_factors(const viennacl::vector_base<T>& classes, const viennacl::vector_base<T>& prod_result, viennacl::vector_base<T>& factors, const bool is_nominal, int loss, double learning_rate, double bias)
	{
		viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(classes).context());

		sgd_kernels<double>::init(ctx);
		static viennacl::ocl::kernel& sgd_map_prod_value = ctx.get_kernel(sgd_kernels<T>::program_name(), "sgd_map_prod_value");
		sgd_map_prod_value.local_work_size(0, 256);
		sgd_map_prod_value.global_work_size(0, 16384);


		//(int N, bool nominal,int loss, double learning_rate, double bias, __global double* class_values, __global double* prod_values, __global double* factor)
		viennacl::ocl::enqueue(sgd_map_prod_value(classes.size(), (cl_bool)is_nominal, loss,  learning_rate, bias, classes.handle().opencl_handle(), prod_result.handle().opencl_handle(),
				factors.handle().opencl_handle()
				));

	};

	template <typename T>
	void sgd_update_weights(viennacl::vector_base<T>& weights, const viennacl::compressed_matrix<T>& batch, const viennacl::vector_base<T>& factors)
	{
		viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(weights).context());
		sgd_kernels<double>::init(ctx);
		static viennacl::ocl::kernel& update_by_factor_kernel = ctx.get_kernel(sgd_kernels<T>::program_name(), "sgd_update_weights");
		update_by_factor_kernel.local_work_size(0, 256);
		update_by_factor_kernel.global_work_size(0, 16384);
		//double* elements,double * factors, int* rows, int * columns)
		viennacl::ocl::enqueue(update_by_factor_kernel(batch.size1(), batch.handle().opencl_handle(), factors.handle().opencl_handle(), batch.handle1().opencl_handle(),
				batch.handle2().opencl_handle()));

		/*viennacl::vector<T> intermediate_sum(weights.size());
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
		weights += intermediate_sum;*/

	};
}

}
}



#endif /* VIENNACL_ML_WEIGHTS_UPDATE_HPP_ */
