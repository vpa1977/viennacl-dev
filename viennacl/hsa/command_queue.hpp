#ifndef VIENNACL_HSA_COMMAND_QUEUE_HPP_
#define VIENNACL_HSA_COMMAND_QUEUE_HPP_

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

/** @file viennacl/hsa/command_queue.hpp
    @brief Implementations of command queue representations
 */


#include <hsa.h>

#include <vector>
#include <string>
#include <sstream>
#include "viennacl/hsa/device.hpp"
#include "viennacl/hsa/handle.hpp"

#include <sys/types.h>

namespace viennacl
{
  namespace hsa
  {

    /** @brief A class representing a command queue
     *
     */
    class command_queue
    {
    public:

      command_queue() : last_index_(0)
      {
      }

      command_queue(const viennacl::hsa::handle<hsa_queue_t*>& h) : handle_(h), last_index_(0)
      {
      }

      //Copy constructor:

      command_queue(command_queue const & other)
      {
        handle_ = other.handle_;
        last_index_ = other.last_index_;
      }

      //assignment operator:

      command_queue & operator=(command_queue const & other)
      {
        handle_ = other.handle_;
        return *this;
      }

      bool operator==(command_queue const & other) const
      {
        return handle_ == other.handle_;
      }

      /** @brief Waits until all kernels in the queue have finished their execution */
      void finish()
      {
#ifdef VIENNACL_HSA_WAIT_KERNEL
    	  	 if (true) return;
#endif

        hsa_signal_t signal;

#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
        std::cout << "ViennaCL: queue flush " << std::endl;
#endif

        uint64_t index = hsa_queue_load_write_index_relaxed(handle_.get());
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
        std::cout << "Finish default queue index " << index << " last_index " << last_index_ << std::endl;
#endif
        if (index == last_index_)
        	return;
        last_index_ = index;
		const uint32_t queueMask =handle_.get()->size - 1;
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
		std::cout << "Need sync with device" << std::endl;
#endif
        hsa_signal_create(1, 0, NULL, &signal);

        /* Obtain the write index for the command queue for this stream.  */

		/* Define the barrier packet to be at the calculated queue index address.  */
		hsa_barrier_or_packet_t* barrier = &(((hsa_barrier_or_packet_t*)(handle_.get()->base_address))[index&queueMask]);
		memset(barrier, 0, sizeof(hsa_barrier_or_packet_t));
		barrier->header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
		barrier->header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
		barrier->header |= 1 << HSA_PACKET_HEADER_BARRIER;
		barrier->header |= HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
		barrier->completion_signal = signal;

		/* Increment write index and ring doorbell to dispatch the kernel.  */
		hsa_queue_store_write_index_relaxed(handle_.get(), index+1);
		hsa_signal_store_relaxed(handle_.get()->doorbell_signal, index);

		/* Wait on completion signal til kernel is finished.  */
		hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

        hsa_signal_destroy(signal);

      }

      void submit_barrier() const
      {
#ifdef VIENNACL_HSA_WAIT_KERNEL
    	  	 if (true) return;
#endif

 #if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
		 std::cout << "ViennaCL: queue barrir " << std::endl;
 #endif

		 uint64_t index = hsa_queue_load_write_index_relaxed(handle_.get());
		const uint32_t queueMask =handle_.get()->size - 1;




		 /* Obtain the write index for the command queue for this stream.  */

		/* Define the barrier packet to be at the calculated queue index address.  */
		hsa_barrier_or_packet_t* barrier = &(((hsa_barrier_or_packet_t*)(handle_.get()->base_address))[index&queueMask]);
		memset(barrier, 0, sizeof(hsa_barrier_or_packet_t));
		barrier->header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
		barrier->header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
		barrier->header |= 1 << HSA_PACKET_HEADER_BARRIER;
		barrier->header |= HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
//		barrier->completion_signal = -1;

		/* Increment write index and ring doorbell to dispatch the kernel.  */
		hsa_queue_store_write_index_relaxed(handle_.get(), index+1);
		hsa_signal_store_relaxed(handle_.get()->doorbell_signal, index);

	   }

      /** @brief Waits until all kernels in the queue have started their execution */
      void flush()
      {
    	  finish();
      }

      viennacl::hsa::handle<hsa_queue_t*> const & handle() const
      {
        return handle_;
      }

      viennacl::hsa::handle<hsa_queue_t*> & handle()
      {
        return handle_;
      }

    private:
      uint64_t last_index_;
      viennacl::hsa::handle<hsa_queue_t*> handle_;
    };

  } //namespace hsa
} //namespace viennacl

#endif
