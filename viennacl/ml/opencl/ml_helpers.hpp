/*
 * weights_update.hpp
 *
 *  Created on: 7/05/2015
 *      Author: bsp
 */

#ifndef VIENNACL_ML_OPENCL_ML_HELPERS_HPP_
#define VIENNACL_ML_OPENCL_ML_HELPERS_HPP_

#include "viennacl/forwards.h"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/handle.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/opencl/common.hpp"
#include "viennacl/ml/opencl/ml_kernels.hpp"
#include "viennacl/ml/knn_sliding_window.hpp"


namespace viennacl
{
namespace ml
{
namespace opencl
{
	template <typename T>
	double reduce( viennacl::vector_base<T>& to_reduce)
	{
		viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(to_reduce).context());
		ml_helper_kernels::init(ctx);
		viennacl::vector<T> reduce_result(1, ctx);
		static viennacl::ocl::kernel& reduce = ctx.get_kernel(ml_helper_kernels::program_name(), "reduce");
		reduce.local_work_size(0,256);
		reduce.global_work_size(0, 6 * 256* 6);
		viennacl::ocl::enqueue(reduce(to_reduce.size(), viennacl::ocl::local_mem(reduce.local_work_size(0) * sizeof(cl_double)),   to_reduce, reduce_result));
		return reduce_result(0);

	}

	template <typename T>
	void sgd_compute_factors(const viennacl::vector_base<T>& classes, const viennacl::vector_base<T>& prod_result, viennacl::vector_base<T>& factors, const bool is_nominal, int loss, double learning_rate, double bias)
	{
		viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(classes).context());

		ml_helper_kernels::init(ctx);
		static viennacl::ocl::kernel& sgd_map_prod_value = ctx.get_kernel(ml_helper_kernels::program_name(), "sgd_map_prod_value");
		sgd_map_prod_value.local_work_size(0, 256);
		sgd_map_prod_value.global_work_size(0, 6 * 256 * 6);


		//(int N, bool nominal,int loss, double learning_rate, double bias, __global double* class_values, __global double* prod_values, __global double* factor)
		cl_ulong size = classes.size();
		viennacl::ocl::enqueue(sgd_map_prod_value(size, (cl_uint)is_nominal,(cl_uint) loss, (cl_double) learning_rate,  (cl_double)bias, classes.handle().opencl_handle(), prod_result.handle().opencl_handle(),
				factors.handle().opencl_handle()
				));

	};

	template <typename sgd_matrix_type, typename T =double>
	void sgd_update_weights(viennacl::vector_base<T>& weights, const sgd_matrix_type& batch, const viennacl::vector_base<T>& factors)
	{
		viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(weights).context());
		ml_helper_kernels::init(ctx);
		static viennacl::ocl::kernel& update_by_factor_kernel = ctx.get_kernel(ml_helper_kernels::program_name(), "sgd_update_weights");
		update_by_factor_kernel.local_work_size(0, 256);
		update_by_factor_kernel.global_work_size(0, 6 * 256 * 6);
		int columns = batch.size2();
		viennacl::vector<int> locks(columns, ctx);
		//double* elements,double * factors, int* rows, int * columns)
		viennacl::ocl::enqueue(update_by_factor_kernel(
				batch.size1(), // rows
				batch.handle().opencl_handle(), // data
				factors.handle().opencl_handle(), // factors vector
				batch.handle1().opencl_handle(), // row indices vector
				batch.handle2().opencl_handle(), // column indices vector
				locks.handle().opencl_handle(), // atomic locks for column access
				weights.handle().opencl_handle() // weights vector
				));

	};

	struct knn_kernels
	{


		static void calc_distance(viennacl::vector<double>& result,const viennacl::ml::knn::dense_sliding_window& sliding_window, int start_row, int end_row, const viennacl::vector<double>& sample)
		{
			const viennacl::matrix<double> samples = sliding_window.m_values_window;
			const viennacl::vector<int> types =  sliding_window.m_attribute_types;
			viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(samples).context());

			ml_helper_kernels::init(ctx);

			static viennacl::ocl::kernel& calc_distance_kernel = ctx.get_kernel(ml_helper_kernels::program_name(), "knn_calc_distance");

			calc_distance_kernel.local_work_size(0, 256);
			calc_distance_kernel.global_work_size(0, 6 * 256 * 6);

			//double* elements,double * factors, int* rows, int * columns)
			viennacl::ocl::enqueue(calc_distance_kernel(
					start_row, // rows
					end_row,
					samples,
					types,
					sample, // row indices vector
					result
					));


		}
	};
}

}
}



#endif /* VIENNACL_ML_WEIGHTS_UPDATE_HPP_ */
