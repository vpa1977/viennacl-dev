/*
 * weights_update.hpp
 *
 *  Created on: 7/05/2015
 *      Author: bsp
 */

#ifndef VIENNACL_ML_WEIGHTS_UPDATE_HPP_
#define VIENNACL_ML_WEIGHTS_UPDATE_HPP_

#include "viennacl/forwards.h"
#include "viennacl/vector.hpp"
#include "viennacl/compressed_matrix.hpp"
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

#include "viennacl/ml/host/weights_update.hpp"

#ifdef VIENNACL_WITH_HSA
#include "viennacl/ml/hsa/ml_helpers.hpp"
#endif

#ifdef VIENNACL_WITH_OPENCL
#include <viennacl/ml/opencl/ml_helpers.hpp>
#endif


namespace viennacl
{
namespace ml
{

	template <typename T>
	void sgd_compute_factors(const  viennacl::vector_base<T>& classes, const viennacl::vector_base<T>& prod_result, viennacl::vector_base<T>& factors, bool is_nominal, int loss, double learning_rate, double bias)
	{
		 switch (viennacl::traits::handle(classes).get_active_handle_id())
		 {
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
        	viennacl::ml::opencl::sgd_compute_factors(classes, prod_result, factors, is_nominal, loss, learning_rate, bias);
          break;
#endif
#ifdef VIENNACL_WITH_HSA
        case viennacl::HSA_MEMORY:
        	viennacl::ml::hsa::sgd_compute_factors(classes, prod_result, factors, is_nominal, loss, learning_rate, bias);
          break;
#endif

        default:
        	throw memory_exception("not implemented");
		 }

	}

	template <typename T=double>
	double sgd_reduce( viennacl::vector_base<T>& v)
	{
	      switch (viennacl::traits::handle(v).get_active_handle_id())
	      {
	        case viennacl::MAIN_MEMORY:
	        {

	        	double result = 0;
	        	double * v_ptr =
	        					viennacl::linalg::host_based::detail::extract_raw_pointer<T>(v);
	        	size_t size = viennacl::traits::size(v);
	        	size_t start = viennacl::traits::start(v);
	        	size_t stride = viennacl::traits::stride(v);
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for if (size > VIENNACL_OPENMP_VE	CTOR_MIN_SIZE)
#endif
	        	for (long i = 0; i < static_cast<long>(size); ++i) {
	        		size_t offset = i * stride + start;
	        		result += v_ptr[offset];
	        	}
	        	return result;
	        }
	          break;
	#ifdef VIENNACL_WITH_OPENCL
	        case viennacl::OPENCL_MEMORY:
	        	return viennacl::ml::opencl::reduce(v);
	          break;
	#endif
	#ifdef VIENNACL_WITH_HSA
	        case viennacl::HSA_MEMORY:
	        	return viennacl::ml::hsa::reduce(v);
	          break;
	#endif


	#ifdef VIENNACL_WITH_CUDA
	        case viennacl::CUDA_MEMORY:
	        	 throw memory_exception("not implemented");
	          break;
	#endif
	        case viennacl::MEMORY_NOT_INITIALIZED:
	          throw memory_exception("not initialised!");
	        default:
	          throw memory_exception("not implemented");
	      }
	}

	template <typename sgd_matrix_type,typename T=double>
	void sgd_update_weights( viennacl::vector_base<T>& weights, const sgd_matrix_type& batch, const viennacl::vector_base<T>& factors)
	{
	      switch (viennacl::traits::handle(weights).get_active_handle_id())
	      {
	        case viennacl::MAIN_MEMORY:
	          viennacl::ml::host::sgd_update_weights(weights, batch, factors);
	          break;
	#ifdef VIENNACL_WITH_OPENCL
	        case viennacl::OPENCL_MEMORY:
	        	viennacl::ml::opencl::sgd_update_weights(weights,batch,factors);
	          break;
	#endif
	#ifdef VIENNACL_WITH_HSA
	        case viennacl::HSA_MEMORY:
	        	viennacl::ml::hsa::sgd_update_weights(weights,batch,factors);
	          break;
	#endif

	#ifdef VIENNACL_WITH_CUDA
	        case viennacl::CUDA_MEMORY:
	        	 throw memory_exception("not implemented");
	          break;
	#endif
	        case viennacl::MEMORY_NOT_INITIALIZED:
	          throw memory_exception("not initialised!");
	        default:
	          throw memory_exception("not implemented");
	      }

	}

}
}




#endif /* VIENNACL_ML_WEIGHTS_UPDATE_HPP_ */
