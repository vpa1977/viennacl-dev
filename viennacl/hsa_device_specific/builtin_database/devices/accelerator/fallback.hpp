#ifndef VIENNACL_HSA_DEVICE_SPECIFIC_BUILTIN_DATABASE_DEVICES_ACCELERATOR_FALLBACK_HPP_
#define VIENNACL_HSA_DEVICE_SPECIFIC_BUILTIN_DATABASE_DEVICES_ACCELERATOR_FALLBACK_HPP_

#include "viennacl/hsa_device_specific/forwards.h"
#include "viennacl/hsa_device_specific/builtin_database/common.hpp"

#include "viennacl/hsa_device_specific/templates/vector_axpy_template.hpp"
#include "viennacl/hsa_device_specific/templates/reduction_template.hpp"
#include "viennacl/hsa_device_specific/templates/matrix_axpy_template.hpp"
#include "viennacl/hsa_device_specific/templates/row_wise_reduction_template.hpp"
#include "viennacl/hsa_device_specific/templates/matrix_product_template.hpp"

namespace viennacl{
namespace hsa_specific{
namespace builtin_database{
namespace devices{
namespace accelerator{
namespace fallback{

inline void add_4B(database_type<vector_axpy_template::parameters_type> & db)
{
  db.add_4B(ocl::unknown_id, CL_DEVICE_TYPE_ACCELERATOR, ocl::unknown, "", vector_axpy_template::parameters_type(1,128,128,FETCH_FROM_GLOBAL_CONTIGUOUS));
}

inline void add_4B(database_type<reduction_template::parameters_type> & db)
{
  db.add_4B(ocl::unknown_id, CL_DEVICE_TYPE_ACCELERATOR, ocl::unknown, "", reduction_template::parameters_type(1,128,128,FETCH_FROM_GLOBAL_CONTIGUOUS));
}

inline void add_4B(database_type<matrix_axpy_template::parameters_type> & db)
{
  db.add_4B(ocl::unknown_id, CL_DEVICE_TYPE_ACCELERATOR, ocl::unknown, "", matrix_axpy_template::parameters_type(1,8,8,8,8,FETCH_FROM_GLOBAL_CONTIGUOUS));
}

inline void add_4B(database_type<row_wise_reduction_template::parameters_type> & db, char_to_type<'N'>)
{
  db.add_4B(ocl::unknown_id, CL_DEVICE_TYPE_ACCELERATOR, ocl::unknown, "", row_wise_reduction_template::parameters_type(1,1,128,128,FETCH_FROM_GLOBAL_CONTIGUOUS));
}

inline void add_4B(database_type<row_wise_reduction_template::parameters_type> & db, char_to_type<'T'>)
{
  db.add_4B(ocl::unknown_id, CL_DEVICE_TYPE_ACCELERATOR, ocl::unknown, "", row_wise_reduction_template::parameters_type(1,1,128,128,FETCH_FROM_GLOBAL_CONTIGUOUS));
}

inline void add_4B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'N'>, char_to_type<'N'>)
{
  db.add_4B(ocl::unknown_id, CL_DEVICE_TYPE_ACCELERATOR, ocl::unknown, "", matrix_product_template::parameters_type(1,8,8,8,4,4,4,FETCH_FROM_LOCAL,FETCH_FROM_LOCAL,8,8));
}

inline void add_4B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'T'>, char_to_type<'N'>)
{
  db.add_4B(ocl::unknown_id, CL_DEVICE_TYPE_ACCELERATOR, ocl::unknown, "", matrix_product_template::parameters_type(1,8,8,8,4,4,4,FETCH_FROM_LOCAL,FETCH_FROM_LOCAL,8,8));
}

inline void add_4B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'N'>, char_to_type<'T'>)
{
  db.add_4B(ocl::unknown_id, CL_DEVICE_TYPE_ACCELERATOR, ocl::unknown, "", matrix_product_template::parameters_type(1,8,8,8,4,4,4,FETCH_FROM_LOCAL,FETCH_FROM_LOCAL,8,8));
}

inline void add_4B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'T'>, char_to_type<'T'>)
{
  db.add_4B(ocl::unknown_id, CL_DEVICE_TYPE_ACCELERATOR, ocl::unknown, "", matrix_product_template::parameters_type(1,8,8,8,4,4,4,FETCH_FROM_LOCAL,FETCH_FROM_LOCAL,8,8));
}


inline void add_8B(database_type<vector_axpy_template::parameters_type> & db)
{
  db.add_8B(ocl::unknown_id, CL_DEVICE_TYPE_ACCELERATOR, ocl::unknown, "", vector_axpy_template::parameters_type(1,128,128,FETCH_FROM_GLOBAL_CONTIGUOUS));
}

inline void add_8B(database_type<reduction_template::parameters_type> & db)
{
  db.add_8B(ocl::unknown_id, CL_DEVICE_TYPE_ACCELERATOR, ocl::unknown, "", reduction_template::parameters_type(1,128,128,FETCH_FROM_GLOBAL_CONTIGUOUS));
}

inline void add_8B(database_type<matrix_axpy_template::parameters_type> & db)
{
  db.add_8B(ocl::unknown_id, CL_DEVICE_TYPE_ACCELERATOR, ocl::unknown, "", matrix_axpy_template::parameters_type(1,8,8,8,8,FETCH_FROM_GLOBAL_CONTIGUOUS));
}

inline void add_8B(database_type<row_wise_reduction_template::parameters_type> & db, char_to_type<'N'>)
{
  db.add_8B(ocl::unknown_id, CL_DEVICE_TYPE_ACCELERATOR, ocl::unknown, "", row_wise_reduction_template::parameters_type(1,1,128,128,FETCH_FROM_GLOBAL_CONTIGUOUS));
}

inline void add_8B(database_type<row_wise_reduction_template::parameters_type> & db, char_to_type<'T'>)
{
  db.add_8B(ocl::unknown_id, CL_DEVICE_TYPE_ACCELERATOR, ocl::unknown, "", row_wise_reduction_template::parameters_type(1,1,128,128,FETCH_FROM_GLOBAL_CONTIGUOUS));
}

inline void add_8B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'N'>, char_to_type<'N'>)
{
  db.add_8B(ocl::unknown_id, CL_DEVICE_TYPE_ACCELERATOR, ocl::unknown, "", matrix_product_template::parameters_type(1,8,8,8,4,4,4,FETCH_FROM_LOCAL,FETCH_FROM_LOCAL,8,8));
}

inline void add_8B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'T'>, char_to_type<'N'>)
{
  db.add_8B(ocl::unknown_id, CL_DEVICE_TYPE_ACCELERATOR, ocl::unknown, "", matrix_product_template::parameters_type(1,8,8,8,4,4,4,FETCH_FROM_LOCAL,FETCH_FROM_LOCAL,8,8));
}

inline void add_8B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'N'>, char_to_type<'T'>)
{
  db.add_8B(ocl::unknown_id, CL_DEVICE_TYPE_ACCELERATOR, ocl::unknown, "", matrix_product_template::parameters_type(1,8,8,8,4,4,4,FETCH_FROM_LOCAL,FETCH_FROM_LOCAL,8,8));
}

inline void add_8B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'T'>, char_to_type<'T'>)
{
  db.add_8B(ocl::unknown_id, CL_DEVICE_TYPE_ACCELERATOR, ocl::unknown, "", matrix_product_template::parameters_type(1,8,8,8,4,4,4,FETCH_FROM_LOCAL,FETCH_FROM_LOCAL,8,8));
}


}
}
}
}
}
}


#endif
