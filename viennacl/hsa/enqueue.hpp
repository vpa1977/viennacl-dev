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

	// get command queue from context
	hsa_queue_t* command_queue = queue.handle().get();
	hsa_dispatch_packet_t aql;
	// create a dispatch packet
	memset(&aql, 0, sizeof(aql));

	// setup dispatch sizes
	aql.dimensions = 1;
	aql.workgroup_size_x = kernel.local_work_size(0);
	aql.grid_size_x = kernel.global_work_size(0);
	if (kernel.global_work_size(1) >= 1) {
		++aql.dimensions;
		aql.grid_size_y = kernel.global_work_size(1);
		aql.workgroup_size_y = kernel.local_work_size(1);
	}
	if (kernel.global_work_size(2) >= 1) {
		++aql.dimensions;
		aql.grid_size_z = kernel.global_work_size(2);
		aql.workgroup_size_z = kernel.local_work_size(2);
	}

	// set dispatch fences
	aql.header.type = HSA_PACKET_TYPE_DISPATCH;
	aql.header.acquire_fence_scope = 2;
	aql.header.release_fence_scope = 2;
	aql.header.barrier = 1;

	// bind kernel code
	aql.kernel_object_address = kernel.handle_->code.handle;
	kernel.arg_buffer_.finalize(kernel.handle_);
	aql.kernarg_address = (uint64_t)kernel.arg_buffer_.kernargs();
	// Initialize memory resources needed to execute
	aql.group_segment_size = kernel.handle_->workgroup_group_segment_byte_size
			+ kernel.arg_buffer_.dynamic_local_size();
	aql.private_segment_size =
			kernel.handle_->workitem_private_segment_byte_size;

	// write packet
	uint32_t queueMask = command_queue->size - 1;
	uint64_t index = hsa_queue_load_write_index_relaxed(command_queue);
	((hsa_dispatch_packet_t*) (command_queue->base_address))[index & queueMask] =
			aql;
	hsa_queue_store_write_index_relaxed(command_queue, index + 1);

	//printf("ring door bell\n");

	// Ring door bell
	hsa_signal_store_relaxed(command_queue->doorbell_signal, index + 1);
} //enqueue()

/** @brief Convenience function that enqueues the provided kernel into the first queue of the currently active device in the currently active context */
template<typename KernelType>
void enqueue(KernelType & k) {
	enqueue(k, k.context().get_queue());
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
