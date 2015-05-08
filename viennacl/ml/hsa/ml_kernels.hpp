/*
 * ml_kernels.hpp
 *
 *  Created on: 7/05/2015
 *      Author: bsp
 */

#ifndef VIENNACL_ML_HSA_ML_KERNELS_HPP_
#define VIENNACL_ML_HSA_ML_KERNELS_HPP_

#include <viennacl/hsa/context.hpp>

namespace viennacl
{
	namespace ml
	{
		namespace hsa
		{
			template< typename NumericT>
			struct sgd_kernels
			{
				static std::string program_name()
				{
				    return "sgd_operations";
				}

				static void init(viennacl::hsa::context& ctx)
				{
					static bool done = false; // TODO : multiple hsa contexts. At the moment there can be only one
					if (done)
						return;
					done = true;
					std::string code;
					code.reserve(1024);

					// for row = [0.. row_count] - update rows with factor results
					const char* const sparse_matrix_by_constant =
							"__kernel sgd_update_weights(double* elements,double * factors, int* rows, int * columns)\
							{\
								int start = rows[ get_global_id() ];\
								int end = rows[ get_global_id() +1];\
								for (int i = start; i < end; ++i)\
								{\
									elements[i] = elements[i] * factors[get_global_id(0)];\
								}\
							}\n";

					code.append(sparse_matrix_by_constant);

					ctx.add_program(code, program_name());

				}
			};
		}

	}
}




#endif /* VIENNACL_ML_HSA_ML_KERNELS_HPP_ */
