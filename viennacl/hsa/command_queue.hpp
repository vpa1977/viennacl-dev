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
  command_queue() {}
  command_queue(viennacl::hsa::handle<hsa_queue_t*> h) : handle_(h) {}

  //Copy constructor:
  command_queue(command_queue const & other)
  {
    handle_ = other.handle_;
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
  void finish() const
  {
	  hsa_signal_t signal;

#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
std::cout << "ViennaCL: queue flush "  << std::endl;
#endif


	  hsa_signal_create(1,0,NULL,&signal);

	  uint64_t index = hsa_queue_load_write_index_relaxed(handle_.get());
	  hsa_barrier_or_packet_t barrier;
	  memset(&barrier, 0, sizeof(barrier));
	  barrier.header =  HSA_PACKET_TYPE_BARRIER_AND;
	  barrier.header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
	  barrier.header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
	  barrier.completion_signal = signal;
	  //barrier.dep_signal[0] = signal_dep;
	  const uint32_t queue_mask = handle_.get()->size - 1;
	  ((hsa_barrier_or_packet_t*)(handle_.get()->base_address))[index&queue_mask]=barrier;
	  hsa_queue_store_write_index_relaxed(handle_.get(), index+1);
	  hsa_signal_store_relaxed(handle_.get()->doorbell_signal, index);
	  hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_LT, 1, (uint64_t) -1, HSA_WAIT_STATE_ACTIVE);
	  hsa_signal_destroy(signal);

  }

  /** @brief Waits until all kernels in the queue have started their execution */
  void flush() const
  {

  }

  viennacl::hsa::handle<hsa_queue_t*> const & handle() const { return handle_; }
  viennacl::hsa::handle<hsa_queue_t*>       & handle()       { return handle_; }

private:

  viennacl::hsa::handle<hsa_queue_t*> handle_;
};

} //namespace hsa
} //namespace viennacl

#endif
