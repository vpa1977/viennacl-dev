#ifndef VIENNACL_HSA_ENQUEUE_HPP_
#define VIENNACL_HSA_ENQUEUE_HPP_

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

/** @file viennacl/hsa/enqueue.hpp
 @brief Enqueues kernels into command queues
 */

#include "viennacl/hsa/backend.hpp"
#include "viennacl/hsa/kernel.hpp"
#include "viennacl/hsa/command_queue.hpp"
#include "viennacl/hsa/context.hpp"

#include <pthread.h>

namespace viennacl
{

  namespace device_specific
  {
    class custom_operation;
    void enqueue_custom_op(viennacl::device_specific::custom_operation & op,
            viennacl::hsa::command_queue const & queue);
  }

  namespace hsa
  {

    /** @brief Enqueues a kernel in the provided queue */
    template<typename KernelType>
    void enqueue(KernelType & kernel, viennacl::hsa::command_queue const & queue)
    {



#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
      std::cout << "ViennaCL: queue" << kernel.name() << " for execution " << std::endl;
#endif

#if defined(VIENNACL_DEBUG_KERNEL_DRYRUN)
      return;
#endif

#if defined(VIENNACL_HSA_WAIT_KERNEL)
      hsa_signal_t signal = queue.completion_signal();
      hsa_signal_store_relaxed(signal,1);
#endif


      // get command queue from context
      hsa_queue_t* command_queue = queue.handle().get();

#ifndef VIENNACL_HSA_WAIT_KERNEL
      const_cast<viennacl::hsa::command_queue&>(queue).sync_queue();
#endif

      // before snack

      /*  Obtain the current queue write index. increases with each call to kernel  */
      uint64_t index = hsa_queue_load_write_index_relaxed(command_queue);
      /* printf("DEBUG:Call #%d to kernel \"%s\" \n",(int) index,"vcopy");  */

      /* Find the queue index address to write the packet info into.  */
      const uint32_t queueMask = command_queue->size - 1;
      hsa_kernel_dispatch_packet_t* this_aql = &(((hsa_kernel_dispatch_packet_t*)(command_queue->base_address))[index&queueMask]);

      /*  FIXME: We need to check for queue overflow here. */

	#if defined(VIENNACL_HSA_WAIT_KERNEL)
      this_aql->completion_signal = signal;
	#else
      this_aql->completion_signal = queue.completion_signal();
	#endif

      /*  Process lparm values */
      /*  this_aql.dimensions=(uint16_t) lparm->ndim; */
      // setup dispatch sizes
      size_t dimensions = 1;
      this_aql->workgroup_size_x = (uint16_t)kernel.local_work_size(0);
      this_aql->grid_size_x = (uint32_t)kernel.global_work_size(0);
      if (kernel.global_work_size(1) >= 1)
      {
        ++dimensions;
        this_aql->grid_size_y = (uint32_t)(kernel.global_work_size(1));
        this_aql->workgroup_size_y = (uint16_t)(kernel.local_work_size(1));
      } else
      {
    	  this_aql->grid_size_y = 1;
    	  this_aql->workgroup_size_y = 1;
      }
      if (kernel.global_work_size(2) >= 1)
      {
        ++dimensions;
        this_aql->grid_size_z = (uint32_t)(kernel.global_work_size(2));
        this_aql->workgroup_size_z = (uint16_t)(kernel.local_work_size(2));
      } else
      {
    	 this_aql->grid_size_z = 1;
    	 this_aql->workgroup_size_z = 1;
      }
      this_aql->setup |= (uint16_t)(dimensions << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS);

      /*  Bind kernel argument buffer to the aql packet.  */
      this_aql->kernel_object = kernel.handle_;
      kernel.arg_buffer_.finalize(kernel.workgroup_group_segment_byte_size_);
      this_aql->kernarg_address = kernel.arg_buffer_.kernargs();
      // Initialize memory resources needed to execute
      this_aql->group_segment_size =(uint32_t)( kernel.workgroup_group_segment_byte_size_
                   + kernel.arg_buffer_.dynamic_local_size());
      this_aql->private_segment_size =(uint32_t)( kernel.workitem_private_segment_byte_size_);


      /*  Prepare and set the packet header */
#ifndef VIENNACL_HSA_WAIT_KERNEL
      this_aql->header |= 1 << HSA_PACKET_HEADER_BARRIER; // use ordered execution
#endif
      this_aql->header |= HSA_FENCE_SCOPE_SYSTEM<< HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE; // scope system - update to define
      this_aql->header |= HSA_FENCE_SCOPE_SYSTEM<< HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
      __atomic_store_n((uint8_t*)(&this_aql->header), (uint8_t)HSA_PACKET_TYPE_KERNEL_DISPATCH, __ATOMIC_RELEASE);

      /* Increment write index and ring doorbell to dispatch the kernel.  */
      hsa_queue_store_write_index_relaxed(command_queue, index+1);




#if defined(VIENNACL_HSA_WAIT_KERNEL)

      hsa_signal_store_relaxed(command_queue->doorbell_signal, index);
      hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_LT, 1, (uint64_t) - 1, HSA_WAIT_STATE_ACTIVE);

     // hsa_signal_destroy(signal);
#else
      const_cast<viennacl::hsa::command_queue&>(queue).dispatch_queue();

#endif        

#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
      std::cout << "ViennaCL: Completed " << kernel.name() << std::endl;
#endif
    } //enqueue()

    /** @brief Convenience function that enqueues the provided kernel into the first queue of the currently active device in the currently active context */
    template<typename KernelType>
    void enqueue(KernelType & k)
    {
      const context& ctx = k.context();
      const command_queue& cmd = ctx.get_queue();
      enqueue(k, cmd);
    }

    inline void enqueue(viennacl::device_specific::custom_operation & op,
            viennacl::hsa::command_queue const & queue)
    {
      device_specific::enqueue_custom_op(op, queue);
    }

    inline void enqueue(viennacl::device_specific::custom_operation & op)
    {
      enqueue(op, viennacl::hsa::current_context().get_queue());
    }

  } // namespace hsa
} // namespace viennacl
#endif
