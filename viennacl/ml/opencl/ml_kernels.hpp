/*
 * ml_kernels.hpp
 *
 *  Created on: 7/05/2015
 *      Author: bsp
 */

#ifndef VIENNACL_ML_OPENCL_ML_KERNELS_HPP_
#define VIENNACL_ML_OPENCL_ML_KERNELS_HPP_

#include <viennacl/ocl/context.hpp>

namespace viennacl
{
	namespace ml
	{
		namespace opencl
		{
			template< typename NumericT>
			struct sgd_kernels
			{
				static std::string program_name()
				{
				    return "sgd_operations";
				}

				static void init(viennacl::ocl::context& ctx)
				{
					static bool done = false; // TODO : multiple hsa contexts. At the moment there can be only one
					if (done)
						return;
					done = true;
					std::string code;
					code.reserve(1024);

					const char* const scan_inclusive =
							"__kernel void scan_inclusive(__global const double* data, __global double* result, uint bin_size) "
							"{ "
							"  uint lid = get_local_id(0); "
							"  uint binId = get_group_id(0); "
							"  uint group_offset = binId * bin_size; "
							"  uint maxval = data[get_local_id(0)]; "
							"  do "
							"  { "
							"		uint binValue = data[group_offset + lid]; "
							"       uint prefix_sum = work_group_scan_exclusive_add( binValue ); "
							"       result[group_offset + lid] = prefix_sum + maxval; "
							"       maxval += work_group_broadcast( prefix_sum + binValue, get_local_size(0)-1 ); "
							"       group_offset += get_local_size(0);"
							"                                         "
							"  } while(group_offset < (binId+1) * bin_size);"
							" }";
				//	code.append(scan_inclusive);

					// for row = [0.. row_count] - update rows with factor results
					const char* const sparse_matrix_by_constant = "\n"
							"__kernel  void sgd_update_weights(int N, __global double* elements,__global double * factors, __global int* rows, __global int * columns)"
							"{"
							"    int id = get_global_id(0);  "
							"    for (; id < N; id+= get_global_size(0)) "
							"    {                                        "
							"    	int start = rows[ id ];              "
								"	int end = rows[ id +1];     "
								"	for (int i = start; i < end; ++i)  "
								"	{                                                                "
								"		elements[i] = elements[i] * factors[id];       "
								"	}                                                    "
							"     }             "
							"}\n";

					code.append(sparse_matrix_by_constant);

					const char* const map_prod_value_nominal ="\n"
							"\n__kernel void sgd_map_prod_value(int N, int nominal,int loss_function, double learning_rate, double bias, __global double* class_values, __global double* prod_values, __global double* factor)\n"
							"{\n"
							"	int id = get_global_id(0);\n"
							"   double y;                 "
							"   double z;                  "
							"	for (; id < N; id+= get_global_size(0))\n "
							"   {														 "
							"       if (nominal)\n"
							"		{	"
							"			y = select(-1,1, (int)class_values[id]); "
							"			z = prod_values[id] * y + bias;"
							"		}"
							"		else\n"
							"		{"
							"			y = class_values[id]; "
							"			z =  y - (prod_values[id] + bias);\n"
							"		}\n"
							"		double loss = 0;   "
							"       switch(loss_function)    "
							"       {"
							"          case 0: loss = isless(z, 1);"
							"				break; "
							"          case 1:"
							"		   {"
							"	        double t = exp(-z); "
							"			loss = fmin(1.0 / (exp(z) + 1.0),t / (t+1)); "
							"		   }"
							"          break;"
							"          case 2:"
							"          {"
							"            loss = z;"
							"          }"
							"          break;"
							"		}"
							"       factor[id] = learning_rate * y * loss;"
							"	}"
							"}\n";
					code.append(map_prod_value_nominal);

					ctx.add_program(code, program_name());


				}
			};
		}

	}
}




#endif /* VIENNACL_ML_HSA_ML_KERNELS_HPP_ */
