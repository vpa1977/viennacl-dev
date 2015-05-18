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
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for if (size > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
		for (long i = 0; i < static_cast<long>(factors.size()); ++i) {
			if (factors(i)) {
				const viennacl::vector<double>& row = viennacl::row(batch, i);
				weights += factors(i) * row;
			}
		}

	}
}

}
}



#endif /* VIENNACL_ML_WEIGHTS_UPDATE_HPP_ */
