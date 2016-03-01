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
#include <deque>
#include <string>
#include <sstream>
#include <atomic>
#include "viennacl/hsa/device.hpp"
#include "viennacl/hsa/handle.hpp"

#include <hsa.h>
#include <hsa_ext_amd.h>
#include <sys/types.h>

#include <pthread.h>

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

//      static bool command_queue_signal_handler(hsa_signal_value_t value, void* arg)
//	  {
//    	 std::cout << "Async callback now " << value << std::endl;

//	  }

      command_queue() : last_index_(0), signal_counter_(0)
      {
    	  pthread_mutex_init(&lock_, NULL);
      }

      command_queue(const viennacl::hsa::handle<hsa_queue_t*>& h) : handle_(h), last_index_(0)
      {
    	  hsa_signal_t signal;
    	  hsa_signal_create(0, 0, NULL, &signal);

    	  /* hsa_status_t status = hsa_amd_signal_async_handler(signal,
    			  HSA_SIGNAL_CONDITION_NE,
    	                               1,
									   &viennacl::hsa::command_queue::command_queue_signal_handler,
									   (void*)this);
    	  if (status != HSA_STATUS_SUCCESS)
    		  throw std::runtime_error("unable to register async helper");*/
    	  completion_signal_ = signal;

    	  last_index_ = hsa_queue_load_write_index_relaxed(handle_.get());
    	  signal_counter_ = 0;
    	  sync_ = true;
    	  pthread_mutex_init(&lock_, NULL);

      }

      virtual ~command_queue()
      {
    	  // move signal to the shared pointer.
    	  pthread_mutex_destroy(&lock_);
      }

      //Copy constructor:

      command_queue(command_queue const & other)
      {
        handle_ = other.handle_;
        last_index_ = other.last_index_;
        completion_signal_ = other.completion_signal_;
        signal_counter_ = other.signal_counter_;
        sync_ = other.sync_;
        m_signal_queue = other.m_signal_queue;
        pthread_mutex_init(&lock_, NULL);
      }

      //assignment operator:

      command_queue & operator=(command_queue const & other)
      {
    	m_signal_queue = other.m_signal_queue;
        handle_ = other.handle_;
        last_index_ = other.last_index_;
        completion_signal_ = other.completion_signal_;
        signal_counter_ = other.signal_counter_;
        sync_ = other.sync_;
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
    	 pthread_mutex_lock(&lock_);

		 if (m_signal_queue.size() )
		 {
			 hsa_signal_t signal = m_signal_queue[m_signal_queue.size() -1];
			 int value = hsa_signal_load_relaxed(signal);
			 if (value == 1)
				 hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_LT, 1, (uint64_t) - 1, HSA_WAIT_STATE_BLOCKED);
		 }
		 pthread_mutex_unlock(&lock_);
      }

      void sync_queue()
      {
#ifdef VIENNACL_HSA_WAIT_KERNEL
    	  	 if (true) return;
#endif
    	  pthread_mutex_lock(&lock_);
    	  pthread_mutex_unlock(&lock_);
      }

      void dispatch_queue(hsa_signal_t signal)
      {
          pthread_mutex_lock(&lock_);
    	  m_signal_queue.push_back(signal);
    	  int i = 0;
    	  for (i = 0 ;i < m_signal_queue.size() && hsa_signal_load_relaxed(signal) < 1; ++i);
    	  if (i > 1)
    		  m_signal_queue.erase(m_signal_queue.begin(), m_signal_queue.begin() + i-1 );

          pthread_mutex_unlock(&lock_);
      /*  uint64_t index = hsa_queue_load_write_index_relaxed(handle_.get());
		const uint32_t queueMask =handle_.get()->size - 1;

		hsa_barrier_and_packet_t* barrier = &(((hsa_barrier_and_packet_t*)(handle_.get()->base_address))[index&queueMask]);
		memset(barrier, 0, sizeof(hsa_barrier_or_packet_t));
		barrier->header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
		barrier->header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
		barrier->header |= 1 << HSA_PACKET_HEADER_BARRIER;
	    barrier->header |= HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;

		hsa_queue_store_write_index_relaxed(handle_.get(), index+1);
		hsa_signal_store_relaxed(handle_.get()->doorbell_signal, index);
*/
      }

      void submit_barrier(int val)
      {
#ifdef VIENNACL_HSA_WAIT_KERNEL
    	  	 if (true) return;
#endif

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

      void count_signals()
      {

      }

      hsa_signal_t completion_signal()
      {
    	  pthread_mutex_lock(&lock_);
    	  bool check = true;
    	  if (m_signal_queue.size())
    	  {
    		  hsa_signal_t s = m_signal_queue.front();
    		  int value = hsa_signal_load_relaxed(s);
    		  if (value < 1)
    		  {
    			  m_signal_queue.pop_front();
    			  hsa_signal_store_relaxed(s, 1);
    			  pthread_mutex_unlock(&lock_);
    			  return s;
    		  }
    	  }
    	  hsa_signal_create(1, 0,NULL,  &completion_signal_);
    	  pthread_mutex_unlock(&lock_);
    	  return completion_signal_;
      }

    private:
      uint64_t last_index_;
      long signal_counter_;
      bool sync_;
      hsa_signal_t completion_signal_;
      std::deque<hsa_signal_t> m_signal_queue;
      viennacl::hsa::handle<hsa_queue_t*> handle_;
      pthread_mutex_t lock_;

    };

  } //namespace hsa
} //namespace viennacl

#endif
