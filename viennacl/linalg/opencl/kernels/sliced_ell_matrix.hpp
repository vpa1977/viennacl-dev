#ifndef VIENNACL_LINALG_OPENCL_KERNELS_SLICED_ELL_MATRIX_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_SLICED_ELL_MATRIX_HPP

/* =========================================================================
   Copyright (c) 2010-2015, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"

#include "viennacl/linalg/opencl/common.hpp"

/** @file viennacl/linalg/opencl/kernels/sliced_ell_matrix.hpp
 *  @brief OpenCL kernel file for sliced_ell_matrix operations */
namespace viennacl
{
namespace linalg
{
namespace opencl
{
namespace kernels
{

//////////////////////////// Part 1: Kernel generation routines ////////////////////////////////////

template<typename StringT>
void generate_sliced_ell_vec_mul(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void vec_mul( \n");
  source.append("  __global const unsigned int * columns_per_block, \n");
  source.append("  __global const unsigned int * column_indices, \n");
  source.append("  __global const unsigned int * block_start, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * x, \n");
  source.append("  uint4 layout_x, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * result, \n");
  source.append("  uint4 layout_result, \n");
  source.append("  unsigned int block_size) \n");
  source.append("{ \n");
  source.append("  uint blocks_per_workgroup = get_local_size(0) / block_size; \n");
  source.append("  uint id_in_block = get_local_id(0) % block_size; \n");
  source.append("  uint num_blocks  = (layout_result.z - 1) / block_size + 1; \n");
  source.append("  uint global_warp_count  = blocks_per_workgroup * get_num_groups(0); \n");
  source.append("  uint global_warp_id     = blocks_per_workgroup * get_group_id(0) + get_local_id(0) / block_size; \n");

  source.append("  for (uint block_idx = global_warp_id; block_idx < num_blocks; block_idx += global_warp_count) { \n");
  source.append("    "); source.append(numeric_string); source.append(" sum = 0; \n");

  source.append("    uint row    = block_idx * block_size + id_in_block; \n");
  source.append("    uint offset = block_start[block_idx]; \n");
  source.append("    uint num_columns = columns_per_block[block_idx]; \n");
  source.append("    for (uint item_id = 0; item_id < num_columns; item_id++) { \n");
  source.append("      uint index = offset + item_id * block_size + id_in_block; \n");
  source.append("      "); source.append(numeric_string); source.append(" val = elements[index]; \n");
  source.append("      sum += val ? (x[column_indices[index] * layout_x.y + layout_x.x] * val) : 0; \n");
  source.append("    } \n");

  source.append("    if (row < layout_result.z) \n");
  source.append("      result[row * layout_result.y + layout_result.x] = sum; \n");
  source.append("  } \n");
  source.append("} \n");
}


//////////////////////////// Part 2: Main kernel class ////////////////////////////////////

// main kernel class
/** @brief Main kernel class for generating OpenCL kernels for ell_matrix. */
template<typename NumericT, typename IndexT, typename Context = viennacl::ocl::context>
struct sliced_ell_matrix;

template<typename NumericT, typename Context>
struct sliced_ell_matrix<NumericT, unsigned int, Context>
{
  static std::string program_name()
  {
    return viennacl::ocl::type_to_string<NumericT>::apply() + viennacl::ocl::type_to_string<unsigned int>::apply() + "_sliced_ell_matrix";
  }

  static void init(Context & ctx)
  {
    static std::map<void*, bool> init_done;
    if (!init_done[ctx.handle().get()])
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<NumericT, Context>::apply(ctx);
      std::string numeric_string = viennacl::ocl::type_to_string<NumericT>::apply();

      std::string source;
      source.reserve(1024);

      viennacl::ocl::append_double_precision_pragma<double>( ctx.current_device().double_support_extension(), source);

      // fully parametrized kernels:
      generate_sliced_ell_vec_mul(source, numeric_string);

      std::string prog_name = program_name();
      #ifdef VIENNACL_BUILD_INFO
      std::cout << "Creating program " << prog_name << std::endl;
      #endif
      ctx.add_program(source, prog_name);
      init_done[ctx.handle().get()] = true;
    } //if
  } //init
};

}  // namespace kernels
}  // namespace opencl
}  // namespace linalg
}  // namespace viennacl
#endif

