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

#define KAVERI_GLOBAL_SIZE (4*6+1)*256

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
		static int num_groups = (ctx.current_device().max_compute_units() * 4 + 1);
		static int  global_size = num_groups * ctx.current_device().max_work_group_size();

		ml_helper_kernels<viennacl::ocl::context>::init(ctx);
		viennacl::vector<T> reduce_result(num_groups, ctx);
		std::vector<T> reduce_result_cpu(num_groups);
		static viennacl::ocl::kernel& reduce = ctx.get_kernel(ml_helper_kernels<viennacl::ocl::context>::program_name(), "reduce");
		reduce.local_work_size(0,ctx.current_device().max_work_group_size());
		reduce.global_work_size(0, global_size);
		viennacl::ocl::enqueue(reduce(to_reduce.size(), viennacl::ocl::local_mem(reduce.local_work_size(0) * sizeof(cl_double)),   to_reduce, reduce_result));
		viennacl::copy(reduce_result, reduce_result_cpu);
		double ret = 0;
		for (auto val : reduce_result_cpu)
			ret += val;
		return ret;

	}

	template <typename T>
	void sgd_compute_factors(const viennacl::vector_base<T>& classes, const viennacl::vector_base<T>& prod_result, viennacl::vector_base<T>& factors, const bool is_nominal, int loss, double learning_rate, double bias)
	{
		viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(classes).context());

		ml_helper_kernels<viennacl::ocl::context>::init(ctx);
		viennacl::ocl::kernel& sgd_map_prod_value = ctx.get_kernel(ml_helper_kernels<viennacl::ocl::context>::program_name(), "sgd_map_prod_value");
		static int  global_size = (ctx.current_device().max_compute_units() *4 +1) * ctx.current_device().max_work_group_size();
		sgd_map_prod_value.local_work_size(0, ctx.current_device().max_work_group_size());
		sgd_map_prod_value.global_work_size(0, global_size);


		//(int N, bool nominal,int loss, double learning_rate, double bias, __global double* class_values, __global double* prod_values, __global double* factor)
		cl_ulong size = classes.size();
		viennacl::ocl::enqueue(sgd_map_prod_value(size, (cl_uint)is_nominal,(cl_uint) loss, (cl_double) learning_rate,  (cl_double)bias, classes.handle().opencl_handle(), prod_result.handle().opencl_handle(),
				factors.handle().opencl_handle()
				));

	};

	template<typename T= double> 
	void dense_sgd_update_weights(int row, bool nominal, double learning_rate, double bias, unsigned int loss_function, const viennacl::vector<double>& class_values, const viennacl::scalar<double>& prod_result, const viennacl::vector<double>& row_values, viennacl::vector<double>& weights)
	{
		viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(weights).context());
		ml_helper_kernels<viennacl::ocl::context>::init(ctx);
		static bool init = false;
		viennacl::ocl::kernel& dense_sgd_map_prod_values_kernel = ctx.get_kernel(ml_helper_kernels<viennacl::ocl::context>::program_name(), "dense_sgd_map_prod_value");
		static int global_size = (ctx.current_device().max_compute_units() * 4 + 1) * ctx.current_device().max_work_group_size();
		dense_sgd_map_prod_values_kernel.local_work_size(0, ctx.current_device().max_work_group_size());
		dense_sgd_map_prod_values_kernel.global_work_size(0, global_size);
		viennacl::ocl::enqueue(dense_sgd_map_prod_values_kernel(
			row_values.size(), // rows
			nominal,
			loss_function,
			row,
			learning_rate,
			bias,
			class_values.handle().opencl_handle(),
			prod_result,
			weights.handle().opencl_handle(), 
			weights.handle().opencl_handle()
			));
		
	}

	template <typename sgd_matrix_type, typename T =double>
	void sgd_update_weights(viennacl::vector_base<T>& weights, const sgd_matrix_type& batch, const viennacl::vector_base<T>& factors)
	{
		viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(weights).context());
		ml_helper_kernels<viennacl::ocl::context>::init(ctx);
		viennacl::ocl::kernel& update_by_factor_kernel = ctx.get_kernel(ml_helper_kernels<viennacl::ocl::context>::program_name(), "sgd_update_weights");
		static int global_size = (ctx.current_device().max_compute_units() *4 +1) * ctx.current_device().max_work_group_size();
		update_by_factor_kernel.local_work_size(0, ctx.current_device().max_work_group_size());
		int work_size = batch.size1();
		if (ctx.current_device().max_work_group_size() >(size_t) work_size)
			work_size = ctx.current_device().max_work_group_size();
		if (work_size > global_size)
			work_size = global_size;
		update_by_factor_kernel.global_work_size(0, work_size);

		/*
                 int columns = batch.size2();
		std::cout << "Factor " << factors(0) << std::endl;
		
		for (int i = 0; i < batch.size2(); ++i)
		{
			double d = ((sgd_matrix_type&)batch)(0, i);
			std::cout << " " << d;
		}
		std::cout << std::endl;
		*/
		viennacl::ocl::enqueue(update_by_factor_kernel(
				batch.size1(), // rows
				batch.handle().opencl_handle(), // data
				factors.handle().opencl_handle(), // factors vector
				batch.handle1().opencl_handle(), // row indices vector 
				batch.handle2().opencl_handle(), // column indices vector
				weights.handle().opencl_handle() // weights vector
				));

/*		std::cout << "Weights after:";
		for (int i = 0; i < weights.size(); ++i)
		{
			std::cout << " " << weights(i);

		}
		std::cout << std::endl;
		std::cout << "matrix after mult ";
		for (int i = 0; i < batch.size2(); ++i)
		{
			double d = ((sgd_matrix_type&)batch)(0, i);
			std::cout << " " << d;
		}
		std::cout << std::endl;
*/
	};

	struct knn_kernels
	{
		static void bitonic_sort(viennacl::vector<double>& result)
		{}


		static void calc_distance(viennacl::vector<double>& result,const viennacl::ml::knn::dense_sliding_window& sliding_window, int start_row, int end_row, const viennacl::vector<double>& sample)
		{
			const viennacl::matrix<double>& samples = sliding_window.m_values_window;
			const viennacl::vector<int>& types =  sliding_window.m_attribute_types;

			viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(samples).context());

			ml_helper_kernels<viennacl::ocl::context>::init(ctx);

			static viennacl::ocl::kernel& calc_distance_kernel = ctx.get_kernel(ml_helper_kernels<viennacl::ocl::context>::program_name(), "knn_calc_distance");
			static int global_size = (ctx.current_device().max_compute_units() *4 +1) * ctx.current_device().max_work_group_size();
			calc_distance_kernel.local_work_size(0, ctx.current_device().max_work_group_size());
			calc_distance_kernel.global_work_size(0,global_size );

			//double* elements,double * factors, int* rows, int * columns)
			viennacl::ocl::enqueue(calc_distance_kernel(
					start_row, // rows
					end_row,
					types.size(),
					samples,
					static_cast<cl_uint>(samples.start1()),
					static_cast<cl_uint>(samples.start2()),
                                        static_cast<cl_uint>(samples.internal_size1()),
					static_cast<cl_uint>(samples.internal_size2()),
					static_cast<cl_uint>(samples.size1()),
					static_cast<cl_uint>(samples.size2()),
					static_cast<cl_uint>(samples.stride1()),
					static_cast<cl_uint>(samples.stride2()),
					sliding_window.m_min_range,
					sliding_window.m_max_range,
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
