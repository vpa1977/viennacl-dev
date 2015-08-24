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

namespace viennacl {

namespace device_specific {
class custom_operation;
void enqueue_custom_op(viennacl::device_specific::custom_operation & op,
		viennacl::hsa::command_queue const & queue);
}

namespace hsa {

/** @brief Enqueues a kernel in the provided queue */
template<typename KernelType>
void enqueue(KernelType & kernel, viennacl::hsa::command_queue const & queue) {



#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
std::cout << "ViennaCL: queue" << kernel.name() << " for execution "  << std::endl;
#endif

#if defined(VIENNACL_DEBUG_KERNEL_DRYRUN)
	return;
#endif

#if defined(VIENNACL_HSA_WAIT_KERNEL)
	hsa_signal_t signal;        
	hsa_signal_create(1,0,NULL,&signal);
#endif
        
	// get command queue from context
	hsa_queue_t* command_queue = queue.handle().get();
	hsa_kernel_dispatch_packet_t aql;
	// create a dispatch packet
	memset(&aql, 0, sizeof(aql));
	aql.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    aql.header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
    aql.header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

	// setup dispatch sizes
	size_t dimensions = 1;
	aql.workgroup_size_x = kernel.local_work_size(0);
	aql.grid_size_x = kernel.global_work_size(0);
	if (kernel.global_work_size(1) >= 1) {
		++dimensions;
		aql.grid_size_y = kernel.global_work_size(1);
		aql.workgroup_size_y = kernel.local_work_size(1);
	}
	else
	{
		aql.grid_size_y = 1;
		aql.workgroup_size_y = 1;
	}
	if (kernel.global_work_size(2) >= 1) {
		++dimensions;
		aql.grid_size_z = kernel.global_work_size(2);
		aql.workgroup_size_z = kernel.local_work_size(2);
	} else
	{
		aql.grid_size_z = 1;
		aql.workgroup_size_z = 1;
	}
	aql.setup  |= dimensions << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;

	// set dispatch fences

	// bind kernel code
	aql.kernel_object = kernel.handle_;
	kernel.arg_buffer_.finalize(kernel.workgroup_group_segment_byte_size_);
	aql.kernarg_address = kernel.arg_buffer_.kernargs();
	// Initialize memory resources needed to execute
	aql.group_segment_size = kernel.workgroup_group_segment_byte_size_
			+ kernel.arg_buffer_.dynamic_local_size();
	aql.private_segment_size =	kernel.workitem_private_segment_byte_size_;
#if defined(VIENNACL_HSA_WAIT_KERNEL)        
	aql.completion_signal = signal;
#endif        
	// write packet
	uint32_t queueMask = command_queue->size - 1;
	uint64_t index = hsa_queue_load_write_index_relaxed(command_queue);
	((hsa_kernel_dispatch_packet_t*) (command_queue->base_address))[index & queueMask] =
			aql;
	hsa_queue_store_write_index_relaxed(command_queue, index + 1);

	//printf("ring door bell\n");

	// Ring door bell
	hsa_signal_store_relaxed(command_queue->doorbell_signal, index );

#if defined(VIENNACL_HSA_WAIT_KERNEL)
	hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_LT, 1, (uint64_t)-1, HSA_WAIT_STATE_ACTIVE);

	hsa_signal_destroy(signal);
#endif        

#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
std::cout << "ViennaCL: Completed " << kernel.name() << std::endl;
#endif
} //enqueue()

/** @brief Convenience function that enqueues the provided kernel into the first queue of the currently active device in the currently active context */
template<typename KernelType>
void enqueue(KernelType & k) {
	const context& ctx = k.context();
	const command_queue& cmd = ctx.get_queue();
	enqueue(k, cmd);
}

inline void enqueue(viennacl::device_specific::custom_operation & op,
		viennacl::hsa::command_queue const & queue) {
	device_specific::enqueue_custom_op(op, queue);
}

inline void enqueue(viennacl::device_specific::custom_operation & op) {
	enqueue(op, viennacl::hsa::current_context().get_queue());
}

} // namespace hsa
} // namespace viennacl
#endif
