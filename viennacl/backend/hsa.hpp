#ifndef VIENNACL_BACKEND_HSA_HPP_
#define VIENNACL_BACKEND_HSAL_HPP_

/* =========================================================================
   Copyright (c) 2010-2014, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/** @file viennacl/backend/opencl.hpp
    @brief Implementations for the OpenCL backend functionality
*/


#include <vector>
#include "viennacl/hsa/handle.hpp"
#include "viennacl/hsa/backend.hpp"

namespace viennacl
{
namespace backend
{
namespace hsa
{

// Requirements for backend:

// * memory_create(size, host_ptr)
// * memory_copy(src, dest, offset_src, offset_dest, size)
// * memory_write_from_main_memory(src, offset, size,
//                                 dest, offset, size)
// * memory_read_to_main_memory(src, offset, size
//                              dest, offset, size)
// *
//

/** @brief Creates an array of the specified size in the current OpenCL context. If the second argument is provided, the buffer is initialized with data from that pointer.
 *
 * @param size_in_bytes   Number of bytes to allocate
 * @param host_ptr        Pointer to data which will be copied to the new array. Must point to at least 'size_in_bytes' bytes of data.
 * @param ctx             Optional context in which the matrix is created (one out of multiple OpenCL contexts, CUDA, host)
 *
 */

inline viennacl::hsa::hsa_registered_pointer memory_create(viennacl::hsa::context const & ctx, vcl_size_t size_in_bytes, const void * host_ptr = NULL)
{
  //std::cout << "Creating buffer (" << size_in_bytes << " bytes) host buffer " << host_ptr << " in context " << &ctx << std::endl;
  return ctx.create_memory_without_smart_handle(static_cast<unsigned int>(size_in_bytes), const_cast<void *>(host_ptr));
};

inline void memory_copy(const viennacl::hsa::handle<viennacl::hsa::hsa_registered_pointer>& src,
										viennacl::hsa::handle<viennacl::hsa::hsa_registered_pointer>& dst,
										size_t src_offset,
										size_t dst_offset,
										size_t bytes_to_copy)
{
	char * ptr1 =(char*)(dst.get().get());
	const char * ptr2 = (const char*)(src.get().get());
	memcpy(  ptr1 + dst_offset,ptr2  + src_offset, bytes_to_copy);
}

inline void memory_write(viennacl::hsa::handle<viennacl::hsa::hsa_registered_pointer>&  dst_buffer,
                         vcl_size_t dst_offset,
                         vcl_size_t bytes_to_copy,
                         const void * ptr,
                         bool /*async*/)
{
  char * dst =(char*)dst_buffer.get().get();
  for (vcl_size_t i=0; i<bytes_to_copy; ++i)
    dst[i+dst_offset] = static_cast<const char *>(ptr)[i];
}

inline void memory_read(const viennacl::hsa::handle<viennacl::hsa::hsa_registered_pointer>&  src_buffer,
                        vcl_size_t src_offset,
                        vcl_size_t bytes_to_copy,
                        void * ptr,
                        bool /*async*/)
{
  const char* src = (const char*)src_buffer.get().get();
  for (vcl_size_t i=0; i<bytes_to_copy; ++i)
    static_cast<char *>(ptr)[i] = src[i+src_offset];
}







}
} //backend
} //viennacl
#endif
