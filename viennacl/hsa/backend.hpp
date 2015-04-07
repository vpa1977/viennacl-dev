#ifndef VIENNACL_HSA_BACKEND_HPP_
#define VIENNACL_HSA_BACKEND_HPP_

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

/** @file viennacl/hsa/backend.hpp
    @brief Implementations of the OpenCL backend, where all contexts are stored in.
*/

#include <vector>
#include "viennacl/hsa/context.hpp"
#include "viennacl/hsa/enqueue.hpp"

namespace viennacl
{
namespace hsa
{

/** @brief A backend that provides contexts for ViennaCL objects (vector, matrix, etc.) */
template<bool dummy = false>  //never use parameter other than default (introduced for linkage issues only)
class backend
{
public:
  /** @brief Switches the current context to the context identified by i
    *
    * @param i   ID of the new active context
    */
  static void switch_context(long i)
  {
    current_context_id_ = i;
  }

  /** @brief Returns the current active context */
  static viennacl::hsa::context & context(long id)
  {
    if (!initialized_[id])
    {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
      std::cout << "ViennaCL: Initializing context no. " << id << std::endl;
#endif

      contexts_[id].init();
      //create one queue per device:
      std::vector<viennacl::hsa::device> devices = contexts_[id].devices();
      for (vcl_size_t j = 0; j<devices.size(); ++j)
        contexts_[id].add_queue(devices[j]);
      initialized_[id] = true;

#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
      std::cout << "ViennaCL: Context no. " << id << " initialized with " << devices.size() << " devices" << std::endl;
      std::cout << "ViennaCL: Device id: " << devices[0].id() << std::endl;
#endif
    }
    return contexts_[id];
  }

  /** @brief Returns the current active context */
  static viennacl::hsa::context & current_context()
  {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Getting current_context with id " << current_context_id_ << std::endl;
#endif
#if defined(VIENNACL_NO_CURRENT_CONTEXT)
    assert(false && bool("ViennaCL: current_context called when disabled"));
#endif
    return backend<dummy>::context(current_context_id_);
  }

  /** @brief Returns the current queue for the active device in the active context */
  static viennacl::hsa::command_queue & get_queue()
  {
    return current_context().get_queue();
  }

  /** @brief Sets a number of devices for the context.
    *
    * @param i    ID of the context to be set up
    * @param devices A vector of OpenCL device-IDs that should be added to the context
    */
  static void setup_context(long i, std::vector<hsa_agent_t> const & devices)
  {
    if (initialized_[i])
      std::cerr << "ViennaCL: Warning in init_context(): Providing a list of devices has no effect, because context for ViennaCL is already created!" << std::endl;
    else
    {
      //set devices for context:
      for (vcl_size_t j = 0; j<devices.size(); ++j)
        contexts_[i].add_device(devices[j]);
    }
  }

  /** @brief Initializes ViennaCL with an already existing context
    *
    * @param i    ID of the context to be set up
    * @param c    The OpenCL handle of the existing context
    * @param devices A vector of OpenCL device-IDs that should be added to the context
    * @param queues   A map of queues for each device
    */
  static void setup_context(long i,
                            hsa_environment c,
                            std::vector<hsa_agent_t> const & devices,
                            std::map< uint64_t, std::vector< hsa_queue_t* > > const & queues)
  {
    assert(devices.size() == queues.size() && bool("ViennaCL expects one queue per device!"));

    if (initialized_[i])
      std::cerr << "ViennaCL: Warning in init_context(): Providing a list of devices has no effect, because context for ViennaCL is already created!" << std::endl;
    else
    {
      //set devices for context:
      for (vcl_size_t j = 0; j<devices.size(); ++j)
        contexts_[i].add_device(devices[j]);

      //init context:
      contexts_[i].init(c);

      //add queues:
      typedef typename std::map< uint64_t, std::vector< hsa_queue_t* > >::const_iterator queue_iterator;
      for (queue_iterator qit = queues.begin();
           qit != queues.end();
           ++qit)
      {
        std::vector<hsa_queue_t*> const & queues_for_device = qit->second;
        for (vcl_size_t j=0; j<queues_for_device.size(); ++j)
          contexts_[i].add_queue(qit->first, queues_for_device[j]);
      }

      initialized_[i] = true;
    }
  }

  /** @brief Initializes ViennaCL with an already existing context
    *
    * @param i    ID of the context to be set up
    * @param c    The OpenCL handle of the existing context
    * @param devices A vector of OpenCL device-IDs that should be added to the context
    * @param queue   One queue per device
    */
  static void setup_context(long i, hsa_environment c, std::vector<hsa_agent_t> const & devices, std::vector<hsa_queue_t*> const & queue)
  {
    assert(devices.size() == queue.size() && bool("ViennaCL expects one queue per device!"));

    //wrap queue vector into map
    std::map< uint64_t , std::vector<hsa_queue_t*> > queues_map;
    for (vcl_size_t j = 0; j<devices.size(); ++j)
      queues_map[devices[j].handle].push_back(queue[j]);

    setup_context(i, c, devices, queues_map);
  }

  /** @brief Add an existing context object to the backend */
  static void add_context(long i, viennacl::hsa::context& c)
  {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Adding context '" << c.handle() << "' as id " << i << std::endl;
    std::cout << "ViennaCL: There are " << c.program_num() << " programs" << std::endl;
#endif
    contexts_[i] = c;
    initialized_[i] = true;
  }

  /** @brief Sets the context device type */
  static void set_context_device_type(long i, hsa_device_type_t t)
  {
    contexts_[i].default_device_type(t);
  }

  /** @brief Sets the maximum number of devices per context. Ignored if a device array is provided as well.  */
  static void set_context_device_num(long i, vcl_size_t num)
  {
    contexts_[i].default_device_num(num);
  }

  /** @brief Sets the context device type */
  static void set_context_platform_index(long i, vcl_size_t pf_index)
  {
    //contexts_[i].platform_index(pf_index);
  }

private:
  static long current_context_id_;
  static std::map<long, bool> initialized_;
  static std::map<long, viennacl::hsa::context> contexts_;
};

template<bool dummy>
long backend<dummy>::current_context_id_ = 0;

template<bool dummy>
std::map<long, bool> backend<dummy>::initialized_;

template<bool dummy>
std::map<long, viennacl::hsa::context> backend<dummy>::contexts_;

////////////////////// current context //////////////////
/** @brief Convenience function for returning the current context */
inline viennacl::hsa::context & current_context()
{
  return viennacl::hsa::backend<>::current_context();
}

/** @brief Convenience function for switching the current context */
inline void switch_context(long i)
{
  viennacl::hsa::backend<>::switch_context(i);
}

/** @brief Convenience function for returning the current context */
inline viennacl::hsa::context & get_context(long i)
{
  return viennacl::hsa::backend<>::context(i);
}

/** @brief Convenience function for setting devices for a context */
inline void setup_context(long i,
                          std::vector<hsa_agent_t> const & devices)
{
  viennacl::hsa::backend<>::setup_context(i, devices);
}

/** @brief Convenience function for setting devices for a context */
inline void setup_context(long i,
                          viennacl::hsa::device const & device)
{
  std::vector<hsa_agent_t> device_id_array(1);
  device_id_array[0] = device.id();
  viennacl::hsa::backend<>::setup_context(i, device_id_array);
}

/** @brief Convenience function for setting up a context in ViennaCL from an existing OpenCL context */
inline void setup_context(long i,
                          hsa_environment c,
                          std::vector<hsa_agent_t> const & devices,
                          std::map< uint64_t, std::vector<hsa_queue_t*> > const & queues)
{
  viennacl::hsa::backend<>::setup_context(i, c, devices, queues);
}

/** @brief Convenience function for setting up a context in ViennaCL from an existing OpenCL context */
inline void setup_context(long i, hsa_environment c, std::vector<hsa_agent_t> const & devices, std::vector<hsa_queue_t*> const & queues)
{
  viennacl::hsa::backend<>::setup_context(i, c, devices, queues);
}

/** @brief Convenience function for setting up a context in ViennaCL from an existing OpenCL context */
inline void setup_context(long i, hsa_environment c, hsa_agent_t d, hsa_queue_t* q)
{
  std::vector<hsa_agent_t> devices(1);
  std::vector<hsa_queue_t*> queues(1);
  devices[0] = d;
  queues[0] = q;
  viennacl::hsa::backend<>::setup_context(i, c, devices, queues);
}

/** @brief Convenience function for setting the default device type for a context */
inline void set_context_device_type(long i, hsa_device_type_t dev_type)
{
  viennacl::hsa::backend<>::set_context_device_type(i, dev_type);
}

/** @brief Convenience function for setting the default device type for a context to GPUs */
inline void set_context_device_type(long i, viennacl::hsa::gpu_tag)
{
  set_context_device_type(i, HSA_DEVICE_TYPE_GPU);
}

/** @brief Convenience function for setting the default device type for a context to CPUs */
inline void set_context_device_type(long i, viennacl::hsa::cpu_tag)
{
  set_context_device_type(i, HSA_DEVICE_TYPE_CPU);
}


/** @brief Convenience function for setting the number of default devices per context */
inline void set_context_device_num(long i, vcl_size_t num)
{
  viennacl::hsa::backend<>::set_context_device_num(i, num);
}


/** @brief Convenience function for setting the platform index
 *
 * @param i         Context ID
 * @param pf_index  The platform index as returned by clGetPlatformIDs(). This is not the ID of type cl_platform_id!
 */
inline void set_context_platform_index(long i, vcl_size_t pf_index)
{
  viennacl::hsa::backend<>::set_context_platform_index(i, pf_index);
}

///////////////////////// get queues ///////////////////
/** @brief Convenience function for getting the default queue for the currently active device in the active context */
inline viennacl::hsa::command_queue & get_queue()
{
  return viennacl::hsa::current_context().get_queue();
}

/** @brief Convenience function for getting the queue for a particular device in the current active context */
inline viennacl::hsa::command_queue & get_queue(viennacl::hsa::device d, unsigned int queue_id = 0)
{
  return viennacl::hsa::current_context().get_queue(d.id(), queue_id);
}

/** @brief Convenience function for getting the queue for a particular device in the current active context */
inline viennacl::hsa::command_queue & get_queue(hsa_agent_t dev_id, unsigned int queue_id = 0)
{
  return viennacl::hsa::current_context().get_queue(dev_id, queue_id);
}


/** @brief Convenience function for getting the kernel for a particular program from the current active context */
inline viennacl::hsa::kernel & get_kernel(std::string const & prog_name, std::string const & kernel_name)
{
  return viennacl::hsa::current_context().get_program(prog_name).get_kernel(kernel_name);
}

/** @brief Convenience function for switching the active device in the current context */
inline void switch_device(viennacl::hsa::device & d)
{
  viennacl::hsa::current_context().switch_device(d);
}

/** @brief Convenience function for returning the active device in the current context */
inline viennacl::hsa::device const & current_device()
{
  return viennacl::hsa::current_context().current_device();
}

} //hsa
} //viennacl
#endif
