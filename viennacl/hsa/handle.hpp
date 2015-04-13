#ifndef VIENNACL_HSA_HANDLE_HPP_
#define VIENNACL_HSA_HANDLE_HPP_

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

/** @file viennacl/hsa/handle.hpp
 @brief Implementation of a smart-pointer-like class for handling hsa handles.
 */

#include <hsa.h>
#include <hsa_ext_finalize.h>
#include <hsa_ext_image.h>
#include <assert.h>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include "viennacl/hsa/forwards.h"
#include "viennacl/hsa/brig_helper.hpp"
#include "viennacl/hsa/device.hpp"
#include "viennacl/hsa/error.hpp"


namespace viennacl {
namespace hsa {

template<class OCL_TYPE>
class handle_release_helper {
	typedef typename OCL_TYPE::ERROR_TEMPLATE_ARGUMENT_FOR_CLASS_INVALID ErrorType;
};

class hsa_environment {
public:
	hsa_environment() {
	}
	void startup() {
		hsa_status_t status;
		status = hsa_init();
		if (status != HSA_STATUS_SUCCESS)
			throw std::runtime_error("cannot init hsa");
		status = hsa_iterate_agents(&hsa_environment::iterate_agent, &devices_);
		if (status != HSA_STATUS_SUCCESS)
			throw std::runtime_error("cannot iterate hsa agents");
	}

	std::vector<device> get_devices(hsa_device_type_t type) const {

		std::vector<device> result;
		std::for_each(devices_.begin(), devices_.end(),
				[&result, type](const device& dev) {
					hsa_device_type_t device_type;
					hsa_agent_get_info(dev.id(), HSA_AGENT_INFO_DEVICE, &device_type);
					if (type == device_type)
					result.push_back(dev);
				});
		return result;

	}

	std::vector<device> get_devices() {
		return devices_;

	}

private:
	static hsa_status_t iterate_agent(hsa_agent_t agent, void* data) {
		((std::vector<device>*) data)->push_back(device(agent));
		return HSA_STATUS_SUCCESS;
	}

	std::vector<device> devices_;

};

class hsa_registered_pointer {
public:
	hsa_registered_pointer(void* ptr, size_t size, bool own = true) {
		m_ptr = ptr;
		m_size = size;
		m_own = own;
	}

	void prepare() {
		hsa_memory_register(m_ptr, m_size);
	}

	void release() {
		hsa_memory_deregister(m_ptr, m_size);
		if (m_own)
		{
			free(m_ptr);
		}
		m_ptr = 0;
	}

	const bool operator==(const hsa_registered_pointer& other) const
	{
		return other.m_ptr == m_ptr;
	}

	void* get() {
		return m_ptr;
	}

	const void* get() const
	{
		return m_ptr;
	}
private:
	bool m_own;
	void * m_ptr;
	size_t m_size;
};

struct hsa_code
{
	hsa_code_object_t code_object_;
	hsa_executable_t executable_;
};

template<>
struct handle_release_helper<hsa_code> {
	static void release(hsa_code& ptr) {
		hsa_executable_destroy( ptr.executable_ );
		hsa_code_object_destroy( ptr.code_object_ );

	}
};

template<>
struct handle_release_helper<hsa_environment> {
	static void release(hsa_environment& ptr) {
		hsa_shut_down();
	}
};

/** \cond */
//hsa_registered_pointer:
template<>
struct handle_release_helper<hsa_registered_pointer> {
	static void release(hsa_registered_pointer & ptr) {
		ptr.release();
	}
};

template<>
struct handle_release_helper<hsa_queue_t*> {
	static void release(hsa_queue_t* const & h) {
		hsa_queue_destroy(h);
	}
};


template<class T>
struct wrapper {
	wrapper(const T& something) :
			m_x(something), m_count(0) {
	}

	T m_x;
	int m_count;
};




/** @brief Handle class the effectively represents a smart pointer for OpenCL handles */
template<class OCL_TYPE>
class handle {

public:
	handle() :
			h_(0), p_context_(NULL) {
	}
	handle(const OCL_TYPE something, const viennacl::hsa::context & c) :
			h_(new wrapper<OCL_TYPE>(something)), p_context_(&c)
	{
		inc();
	}
	handle(const handle & other) :
			h_(other.h_), p_context_(other.p_context_) {
		if (h_ != 0)
			inc();
	}
	~handle() {
		if (h_ != 0)
			dec();
	}

	size_t refcount()
	{
		if (h_)
			return h_->m_count;
		return 0;
	}

	/** @brief Copies the OpenCL handle from the provided handle. Does not take ownership like e.g. std::auto_ptr<>, so both handle objects are valid (more like shared_ptr). */
	handle & operator=(const handle & other) {
		if (other.h_ == h_)
			return *this;
		if (h_ != 0)
			dec();
		h_ = other.h_;
		p_context_ = other.p_context_;
		inc();
		return *this;
	}

	/** @brief Wraps an OpenCL handle. Does not change the context of this handle object! Decreases the reference count if the handle object is destroyed or another OpenCL handle is assigned. */
	handle & operator=(const OCL_TYPE & something) {
		if (h_ != 0)
			dec();
		h_ = new wrapper<OCL_TYPE>(something);
		inc();
		return *this;
	}

	/** @brief Wraps an OpenCL handle including its associated context. Decreases the reference count if the handle object is destroyed or another OpenCL handle is assigned. */
	handle & operator=(std::pair<OCL_TYPE, cl_context> p) {
		if (h_ != 0)
			dec();
		h_ = new wrapper<OCL_TYPE>(p.first);
		inc();
		p_context_ = p.second;
		return *this;
	}

	/** @brief Implicit conversion to the plain OpenCL handle. DEPRECATED and will be removed some time in the future. */
	operator OCL_TYPE() const {
		return h_->m_x;
	}

	const OCL_TYPE & get() const {
		return h_->m_x;
	}

	 OCL_TYPE & get()  {
			return h_->m_x;
		}

	viennacl::hsa::context const & context() const {
		assert(
				p_context_ != NULL && bool("Logic error: Accessing dangling context from handle."));
		return *p_context_;
	}
	void context(const viennacl::hsa::context* c) {
		p_context_ =  c;
	}

	/** @brief Swaps the OpenCL handle of two handle objects */
	handle & swap(handle & other) {
		OCL_TYPE tmp = other.h_;
		other.h_ = this->h_;
		this->h_ = tmp;

		viennacl::hsa::context const * tmp2 = other.p_context_;
		other.p_context_ = this->p_context_;
		this->p_context_ = tmp2;

		return *this;
	}

	/** @brief Manually increment the OpenCL reference count. Typically called automatically, but is necessary if user-supplied memory objects are wrapped. */
	void inc() {
		++h_->m_count;
	}
	/** @brief Manually decrement the OpenCL reference count. Typically called automatically, but might be useful with user-supplied memory objects.  */
	void dec() {
		if (--h_->m_count == 0) {
			handle_release_helper<OCL_TYPE>::release(h_->m_x);
			delete h_;
			h_ = 0;
		}
	}
	std::ostream & operator<<(std::ostream &os) {
	    return os << "(handle)";
	}
private:
	wrapper<OCL_TYPE>* h_;
	viennacl::hsa::context const * p_context_;
};


} //namespace hsa
} //namespace viennacl

#endif
