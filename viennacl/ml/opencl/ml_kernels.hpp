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
                    template<class context_class>
			struct ml_helper_kernels
			{
				static std::string program_name()
				{
				    return "ml_helpers";
				}

				static void init(context_class& ctx)
				{
					static bool done = false; // TODO : multiple hsa contexts. At the moment there can be only one
					if (done)
						return;
					done = true;
					std::string code;
					code.reserve(1024);

					const char* const pragmas =
							"\n#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
							"\n#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable\n"
							"\n#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable\n"
							"\n#define WAVEFRONT_SIZE 64\n";
					code.append(pragmas);

					const char* const index =
							"\n#define IDX(row,col,A_start1, A_stride1, A_start2,A_stride2,  A_internal_size2) (A_start1 + A_stride1 * row) * A_internal_size2 + (A_start2 + A_stride2 * col)\n";

					code.append(index);

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
							"\n__kernel void reduce(ulong N, __local double* local_buffer, __global const double* in, __global double* result)"
							"{"
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
							  " 	result[get_group_id(0)] = tmp;"
							  "}"
							"}\n";
					code.append(reduce);

					const char * const atomic_add_helper =
						"\nvoid my_atomic_add(__global double * loc, const double f)\n"
						"\n{															  "
						"	double old = *loc;                                        "
						"	double sum = old + f;                                     "
						"	volatile bool test = true;                                "
						"	while ((test = atomic_compare_exchange_weak((atomic_long*)loc, (long*)&old, (long)sum)) == false) "
						"		sum = old + f; "
						"\n}\n";
					code.append(atomic_add_helper);

					// for row = [0.. row_count] - update rows with factor results
					/*const char* const sparse_matrix_by_constant = "\n"
						"__kernel  void sgd_update_weights(ulong N, __global double* elements,__global double * factors, __global int* rows, __global int * columns,  __global double* output)"
						"{"
						"    int id = get_global_id(0);  "
						"    for (; id < N; id+= get_global_size(0)) "
						"    {                "
						"		if (factors[id] != 0)                "
						"       {										"
						"           int start = rows[ id ];              "
						"			int end = rows[ id +1];     "
						"			for (int i = start; i < end; ++i)  { "
						"				double upd = elements[i] * factors[id]; "
						"               int col = columns[i];  "
						"               my_atomic_add(&output[col], upd);"
						"            }                                                "
						"       }             "
						"    }"
					    "}\n";*/

                                        const char* const sparse_matrix_update_sparse_row =
                                                "\n__kernel void update_sparse_row(int range, __global double* output,__global double* elements, __global int* columns, double factor, int start)"
						"\n{"
                                                " for (int id = get_global_id(0) ; id< range; id+= get_global_size(0)) {"
						"    double upd = elements[start + get_global_id(0)] * factor; int col = columns[ start + get_global_id(0)]; my_atomic_add(&output[col], upd);"
                                                " }"
						"}\n";

                                        code.append(sparse_matrix_update_sparse_row);
					const char* const sparse_matrix_by_constant = "\n"
						"\n__kernel void update_row_weights( __global double* output,__global double* elements, __global int* columns, double factor, int start)"
						"\n{"
						"    double upd = elements[start + get_global_id(0)] * factor; int col = columns[ start + get_global_id(0)]; my_atomic_add(&output[col], upd);"
						"}\n"
						"__kernel  void sgd_update_weights(ulong N, __global double* elements,__global double * factors, __global int* rows, __global int * columns,  __global double* output)"
						"{"
						"    int id = get_global_id(0);  "
						"    for (; id < N; id+= get_global_size(0)) "
						"    {                "
						"		if (factors[id] != 0)                "
						"       {										"

						"           int start = rows[ id ];              "
						"			int end = rows[ id +1];     "
						"           ndrange_t range = ndrange_1D(end - start); "
						"           enqueue_kernel(get_default_queue() , "
						"           CLK_ENQUEUE_FLAGS_NO_WAIT, range,^{update_row_weights(output,elements, columns, factors[id],start); });  "
						"       }             "
						"    }"
						"}\n";

					code.append(sparse_matrix_by_constant);

					const char* const map_prod_value_nominal = "\n"
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
						"			z =  y - (prod_values[id] + bias); y = 1;\n"
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


					const char* const dense_sgd_update_weights =
						"\n__kernel void dense_sgd_map_prod_value(ulong N, uint nominal,uint loss_function, uint row, double learning_rate, double bias, __global double* class_values, double prod_value, __global double* weights, __global double* instance)\n"
						"{\n"
						"	int id = get_global_id(0);\n"
						"   double y, z, factor;                 "
						"	if (id == 0)\n "
						"   {														 "
						"       if (nominal)\n"
						"		{	"
						"			y = select(-1,1, (int)class_values[row]); "
						"			z = y*(prod_value  + bias);"
						"		}"
						"		else\n"
						"		{"
						"			y = class_values[row]; "
						"			z =  y - (prod_value + bias); y = 1;\n"
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
						"       factor = learning_rate * y * loss;"
						"	}"
						"	for (; id < N; id+= get_global_size(0))\n "
						"   {"
						"      weights[id] += factor*instance[id];    "
						"   }"
						"}\n";
					code.append(dense_sgd_update_weights);

					const char* const knn_calc_distance =
							"\n__kernel void knn_calc_distance(int start_row, int end_row, ulong instance_length, __global double* samples,"
							"uint samples_start1, uint samples_start2, uint samples_internal_size1, uint samples_internal_size2, uint samples_size1, "
							"uint samples_size2, uint samples_stride1, uint samples_stride2, "
							" __global double* min, __global double* max, __global int* types, __global double* test, __global double* result)"
							"{\n"
							" 	for (int id = get_global_id(0) + start_row;id < end_row; id += get_global_size(0) )"
							" 	{"
							"       double cur = 0;"
							"		for(int offset = id; offset < id+instance_length ; ++offset)"
							"       {"
							"           int idx = offset -id;"
							"           double diff = max[idx] - min[idx];"
							"           int loc =  IDX(id,idx,samples_start1, samples_stride1, samples_start2, samples_stride2, samples_internal_size2);"
							"			double d =  (samples[loc]  - test[idx])/diff;"
							"			if (types[idx] > 0 ) "
							"				 cur += d != 0 ? 1 : 0;"
							"			else"
							"				 cur += d*d; "
							"		}"
							"		result[id-start_row] = cur; "
							" 	} "
							"}\n" ;

					code.append(knn_calc_distance);



					//code.append(bitonic_sort);

					ctx.add_program(code, program_name());




				}
			};
		}

	}
}




#endif /* VIENNACL_ML_HSA_ML_KERNELS_HPP_ */

