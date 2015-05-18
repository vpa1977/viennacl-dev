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
			struct ml_helper_kernels
			{
				static std::string program_name()
				{
				    return "ml_helpers";
				}

				static void init(viennacl::ocl::context& ctx)
				{
					static bool done = false; // TODO : multiple hsa contexts. At the moment there can be only one
					if (done)
						return;
					done = true;
					std::string code;
					code.reserve(1024);

					const char* const pragmas =
							"\n#pragma OPENCL EXTENSION cl_amd_printf : enable\n"
							"\n#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
							"\n#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable\n"
							"\n#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable\n"
							"\n#define WAVEFRONT_SIZE 64\n";
					code.append(pragmas);


					const char* const scan_inclusive =
							"__kernel void create_histogram(__global const double* data, __global double* result, uint bin_size) "
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
					code.append(scan_inclusive);

					const char* const reduce =
							"\n__global volatile atomic_int global_barrier = ATOMIC_VAR_INIT(0); \n"
							"\n__kernel void reduce(ulong N, __local double* local_buffer, __global const double* in, __global double* result)"
							"{"
							  "int barrier_unset = 0;"
							  // Get our global thread ID
							  "int id = get_global_id(0);"
							  "const int lid = get_local_id(0);"
							  "const int group_size = get_local_size(0);"
							  "local_buffer[lid] = 0;"
							  "for (int idx = id; idx < N; idx+= get_global_size(0))"
							  "  local_buffer[lid]+= in[idx];"
							  "double tmp =  local_buffer[lid];"
							  "barrier(CLK_LOCAL_MEM_FENCE);"
							  // local memory reduction
							  "int i = group_size/2;"
							  "for(; i>WAVEFRONT_SIZE; i >>= 1) {"
								  "if(lid < i)"
									  "local_buffer[lid] = tmp = tmp + local_buffer[lid + i];"
								  "barrier(CLK_LOCAL_MEM_FENCE);"
							  "}"
							  // wavefront reduction
							  "for(; i>0; i >>= 1) {"
							  "	if(lid < i) "
							  " 	local_buffer[lid] = tmp = tmp + local_buffer[lid + i];"
							  "}"

							  "if(lid==0) {"
							  //"     if (get_global_id(0) == 0) result[0] = 0;"
							  " 	while (!atomic_compare_exchange_weak(&global_barrier, &barrier_unset, 1))"
							  "		{ barrier_unset = 0;}"
							  " 	result[0] += tmp;"
							  " 	atomic_exchange(&global_barrier, 0);"
							  "}"
							"}\n";
					code.append(reduce);


					// for row = [0.. row_count] - update rows with factor results
					const char* const sparse_matrix_by_constant = "\n"
							"__kernel  void sgd_update_weights(ulong N, __global double* elements,__global double * factors, __global int* rows, __global int * columns, __global atomic_int* locks, __global double* output)"
							"{"
							"    int barrier_unset = 0; "
							"    int id = get_global_id(0);  "
							"    for (; id < N; id+= get_global_size(0)) "
							"    {                "
							"		if (factors[id] != 0)                "
							"       {										"
							"    		int start = rows[ id ];              "
							"			int end = rows[ id +1];     "
							"			for (int i = start; i < end; ++i)  "
							"			{             "
							"				int idx = columns[i];                          				         "
								"			double upd = elements[i] * factors[id];       "
							"				while (!atomic_compare_exchange_weak(&locks[idx], &barrier_unset, 1)) { barrier_unset = 0; }"
							"				output[idx] += upd;\n"
							"           	atomic_exchange(&locks[idx], 0); "
							"	  	   }                                                    "
							"       }             "
							"    }"
							"}\n";

					code.append(sparse_matrix_by_constant);

					const char* const map_prod_value_nominal ="\n"
							"\n__kernel void sgd_map_prod_value(ulong N, uint nominal,uint loss_function, double learning_rate, double bias, __global double* class_values, __global double* prod_values, __global double* factor)\n"
							"{\n"
							"	int id = get_global_id(0);\n"
							"   double y;                 "
							"   double z;                  "
							"	for (; id < N; id+= get_global_size(0))\n "
							"   {														 "
							"       if (nominal)\n"
							"		{	"
							"			y = select(-1,1, (int)class_values[id]); "
							"			z = y*(prod_values[id]  + bias);"
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
