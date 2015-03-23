#ifndef VIENNACL_HSA_CONTEXT_HPP_
#define VIENNACL_HSA_CONTEXT_HPP_

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

/** @file viennacl/hsa/context.hpp
    @brief Represents an OpenCL context within ViennaCL
*/

#include <hsa.h>
#include <hsa_ext_finalize.h>

#include <algorithm>
#include <fstream>
#include <vector>
#include <map>
#include <cstdlib>
#include "viennacl/hsa/forwards.h"
#include "viennacl/hsa/handle.hpp"
#include "viennacl/hsa/kernel.hpp"
#include "viennacl/hsa/program.hpp"
#include "viennacl/hsa/device.hpp"
#include "viennacl/hsa/command_queue.hpp"
#include "viennacl/hsa/brig_compiler.hpp"
#include "viennacl/hsa/brig_helper.hpp"
#include "viennacl/tools/sha1.hpp"
#include "viennacl/tools/shared_ptr.hpp"

namespace viennacl
{
namespace hsa
{
/** @brief Manages an OpenCL context and provides the respective convenience functions for creating buffers, etc.
  *
  * This class was originally written before the OpenCL C++ bindings were standardized.
  * Regardless, it provides a couple of convience functionality which is not covered by the OpenCL C++ bindings.
*/
class context
{
  typedef std::vector< tools::shared_ptr<viennacl::hsa::program> >   program_container_type;

public:
  context() : initialized_(false),
    device_type_(HSA_DEVICE_TYPE_GPU),
    current_device_id_(0),
    default_device_num_(1),
    pf_index_(0),
    current_queue_id_(0)
  {
    if (std::getenv("VIENNACL_CACHE_PATH"))
      cache_path_ = std::getenv("VIENNACL_CACHE_PATH");
    else
      cache_path_ = "";
  }

  //////// Get and set kernel cache path */
  /** @brief Returns the compiled kernel cache path */
  std::string cache_path() const { return cache_path_; }

  /** @brief Sets the compiled kernel cache path */
  void cache_path(std::string new_path) { cache_path_ = new_path; }

  //////// Get and set default number of devices per context */
  /** @brief Returns the maximum number of devices to be set up for the context */
  vcl_size_t default_device_num() const { return default_device_num_; }

  /** @brief Sets the maximum number of devices to be set up for the context */
  void default_device_num(vcl_size_t new_num) { default_device_num_ = new_num; }

  ////////// get and set preferred device type /////////////////////
  /** @brief Returns the default device type for the context */
  hsa_device_type_t default_device_type()
  {
    return device_type_;
  }

  /** @brief Sets the device type for this context */
  void default_device_type(hsa_device_type_t dtype)
  {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Setting new device type for context " << h_ << std::endl;
#endif
    if (!initialized_)
      device_type_ = dtype; //assume that the user provided a correct value
  }

  //////////////////// get devices //////////////////
  /** @brief Returns a vector with all devices in this context */
  std::vector<viennacl::hsa::device> const & devices() const
  {
    return devices_;
  }

  /** @brief Returns the current device */
  viennacl::hsa::device const & current_device() const
  {
    //std::cout << "Current device id in context: " << current_device_id_ << std::endl;
    return devices_[current_device_id_];
  }

  /** @brief Switches the current device to the i-th device in this context */
  void switch_device(vcl_size_t i)
  {
    assert(i < devices_.size() && bool("Provided device index out of range!"));
    current_device_id_ = i;
  }

  /** @brief If the supplied device is used within the context, it becomes the current active device. */
  void switch_device(viennacl::hsa::device const & d)
  {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Setting new current device for context " << h_ << std::endl;
#endif
    bool found = false;
    for (vcl_size_t i=0; i<devices_.size(); ++i)
    {
      if (devices_[i] == d)
      {
        found = true;
        current_device_id_ = i;
        break;
      }
    }
    if (found == false)
      std::cerr << "ViennaCL: Warning: Could not set device " << d.name() << " for context." << std::endl;
  }

  /** @brief Add a device to the context. Must be done before the context is initialized */
  void add_device(viennacl::hsa::device const & d)
  {
    assert(!initialized_ && bool("Device must be added to context before it is initialized!"));
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Adding new device to context " << h_ << std::endl;
#endif
    if (std::find(devices_.begin(), devices_.end(), d) == devices_.end())
      devices_.push_back(d);
  }

  /** @brief Add a device to the context. Must be done before the context is initialized */
  void add_device(hsa_agent_t d)
  {
    assert(!initialized_ && bool("Device must be added to context before it is initialized!"));
    add_device(viennacl::hsa::device(d));
  }


  /////////////////////// initialize context ///////////////////

  /** @brief Initializes a new context */
  void init()
  {
    init_new();
  }

  /** @brief Initializes the context from an existing, user-supplied context */
  void init(hsa_environment c)
  {
    init_existing(c);
  }

  /*        void existing_context(cl_context context_id)
    {
      assert(!initialized_ && bool("ViennaCL: FATAL error: Provided a new context for an already initialized context."));
      #i#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
      std::cout << "ViennaCL: Reusing existing context " << h_ << std::endl;
      #e#endif
      h_ = context_id;
    }*/

  ////////////////////// create memory /////////////////////////////

  /** @brief Creates a memory buffer within the context. Does not wrap the OpenCL handle into the smart-pointer-like viennacl::hsa::handle, which saves an OpenCL backend call, yet the user has to ensure that the OpenCL memory handle is free'd or passed to a viennacl::hsa::handle later on.
    *
    *  @param flags  OpenCL flags for the buffer creation
    *  @param size   Size of the memory buffer in bytes
    *  @param ptr    Optional pointer to CPU memory, with which the OpenCL memory should be initialized
    *  @return       A plain OpenCL handle. Either assign it to a viennacl::hsa::handle<cl_mem> directly, or make sure that you free to memory manually if you no longer need the allocated memory.
    */
  hsa_registered_pointer create_memory_without_smart_handle(unsigned int size, void * ptr = NULL) const
  {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Creating memory of size " << size << " for context " << h_ << " (unsafe, returning cl_mem directly)" << std::endl;
#endif

    if (!ptr)
    	ptr = malloc(size);
    hsa_registered_pointer hsa_ptr(ptr,size);
    hsa_ptr.prepare();
    return hsa_ptr;
  }


  /** @brief Creates a memory buffer within the context
    *
    *  @param flags  OpenCL flags for the buffer creation
    *  @param size   Size of the memory buffer in bytes
    *  @param ptr    Optional pointer to CPU memory, with which the OpenCL memory should be initialized
    */
  viennacl::hsa::handle<hsa_registered_pointer> create_memory(unsigned int size, void * ptr = NULL) const
  {
    return viennacl::hsa::handle<hsa_registered_pointer>(create_memory_without_smart_handle(size, ptr), *this);
  }

  /** @brief Creates a memory buffer within the context initialized from the supplied data
    *
    *  @param flags  OpenCL flags for the buffer creation
    *  @param buffer A vector (STL vector, ublas vector, etc.)
    */
  template< typename NumericT, typename A, template<typename, typename> class VectorType >
  viennacl::hsa::handle<cl_mem> create_memory(cl_mem_flags flags, const VectorType<NumericT, A> & buffer) const
  {
    return viennacl::hsa::handle<hsa_registered_pointer>(create_memory_without_smart_handle(static_cast<cl_uint>(sizeof(NumericT) * buffer.size()), (void*)&buffer[0]), *this);
  }

  //////////////////// create queues ////////////////////////////////

  /** @brief Adds an existing queue for the given device to the context */
  void add_queue(hsa_agent_t dev, hsa_queue_t* q)
  {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Adding existing queue " << q << " for device " << dev << " to context " << h_ << std::endl;
#endif
    viennacl::hsa::handle<hsa_queue_t*> queue_handle(q, *this);
    queues_[dev].push_back(viennacl::hsa::command_queue(queue_handle));
    queues_[dev].back().handle().inc();
  }

  /** @brief Adds a queue for the given device to the context */
  void add_queue(hsa_agent_t dev)
  {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Adding new queue for device " << dev << " to context " << h_ << std::endl;
#endif
    size_t queue_size = 0;
    hsa_status_t status = hsa_agent_get_info(dev, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
    if (status != HSA_STATUS_SUCCESS)
    	throw std::runtime_error("unable to get queue size");

    hsa_queue_t* command_queue;
    hsa_queue_create(dev, queue_size, HSA_QUEUE_TYPE_MULTI, NULL, NULL, &command_queue);

    viennacl::hsa::command_queue queue(viennacl::hsa::handle<hsa_queue_t*>(command_queue, *this));
    queues_[dev].push_back(queue);
  }

  /** @brief Adds a queue for the given device to the context */
  void add_queue(viennacl::hsa::device d) { add_queue(d.id()); }

  //get queue for default device:
  viennacl::hsa::command_queue & get_queue()
  {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Getting queue for device " << devices_[current_device_id_].name() << " in context " << h_ << std::endl;
    std::cout << "ViennaCL: Current queue id " << current_queue_id_ << std::endl;
#endif

    return queues_[devices_[current_device_id_].id()][current_queue_id_];
  }

  viennacl::hsa::command_queue const & get_queue() const
  {
    typedef std::map< hsa_agent_t, std::vector<viennacl::hsa::command_queue> >    QueueContainer;

#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Getting const queue for device " << devices_[current_device_id_].name() << " in context " << h_ << std::endl;
    std::cout << "ViennaCL: Current queue id " << current_queue_id_ << std::endl;
#endif

    // find queue:
    QueueContainer::const_iterator it = queues_.find(devices_[current_device_id_].id());
    if (it != queues_.end()) {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
      std::cout << "ViennaCL: Queue handle " << (it->second)[current_queue_id_].handle() << std::endl;
#endif
      return (it->second)[current_queue_id_];
    }

    std::cerr << "ViennaCL: FATAL ERROR: Could not obtain current command queue!" << std::endl;
    std::cout << "Number of queues in context: " << queues_.size() << std::endl;
    std::cout << "Number of devices in context: " << devices_.size() << std::endl;
    throw "queue not found!";

    //return ((*it)->second)[current_queue_id_];
  }

  //get a particular queue:
  /** @brief Returns the queue with the provided index for the given device */
  viennacl::hsa::command_queue & get_queue(hsa_agent_t dev, vcl_size_t i = 0)
  {
    if (i >= queues_[dev].size())
      throw invalid_command_queue();

#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Getting queue " << i << " for device " << dev << " in context " << h_ << std::endl;
#endif
    unsigned int device_index;
    for (device_index = 0; device_index < devices_.size(); ++device_index)
    {
      if (devices_[device_index] == dev)
        break;
    }

    assert(device_index < devices_.size() && bool("Device not within context"));

    return queues_[devices_[device_index].id()][i];
  }

  /** @brief Returns the current device */
  // TODO: work out the const issues
  viennacl::hsa::command_queue const & current_queue() //const
  {
    return queues_[devices_[current_device_id_].id()][current_queue_id_];
  }

  /** @brief Switches the current device to the i-th device in this context */
  void switch_queue(vcl_size_t i)
  {
    assert(i < queues_[devices_[current_device_id_].id()].size() && bool("In class 'context': Provided queue index out of range for device!"));
    current_queue_id_ = i;
  }

  /** @brief If the supplied command_queue is used within the context, it becomes the current active command_queue, the command_queue's device becomes current active device. */
  void switch_queue(viennacl::hsa::command_queue const & q)
  {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Setting new current queue for context " << h_ << std::endl;
#endif
    bool found = false;
    typedef std::map< hsa_agent_t, std::vector<viennacl::hsa::command_queue> >    QueueContainer;

    // For each device:
    vcl_size_t j = 0;
    for (QueueContainer::const_iterator it=queues_.begin(); it != queues_.end(); it++,j++)
    {
      const std::vector<viennacl::hsa::command_queue> & qv = (it->second);
      // For each queue candidate
      for (vcl_size_t i=0; i<qv.size(); ++i)
      {
        if (qv[i] == q)
        {
          found = true;
          current_device_id_ = j;
          current_queue_id_ = i;
          break;
        }
      }
    }
    if (found == false)
      std::cerr << "ViennaCL: Warning: Could not set queue " << q.handle().get() << " for context." << std::endl;
  }

  /////////////////// create program ///////////////////////////////
  /** @brief Adds a program to the context
    */
  viennacl::hsa::program & add_program(brig_module p, std::string const & prog_name)
  {
    programs_.push_back(tools::shared_ptr<hsa::program>(new viennacl::hsa::program(p, *this, prog_name)));
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Adding program '" << prog_name << "' with cl_program to context " << h_ << std::endl;
#endif
    return *programs_.back();
  }

  /** @brief Adds a new program with the provided source to the context. Compiles the program and extracts all kernels from it
    */
  viennacl::hsa::program & add_program(std::string const & source, std::string const & prog_name)
  {
    const char * source_text = source.c_str();
    //vcl_size_t source_size = source.size();
    //cl_int err;

#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Adding program '" << prog_name << "' with source to context " << h_ << std::endl;
#endif

    brig_module temp;

    //
    // Retrieves the program in the cache
    //
    if (cache_path_.size())
    {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
      std::cout << "ViennaCL: Cache at " << cache_path_ << std::endl;
#endif

      std::string prefix;
      for(std::vector< viennacl::hsa::device >::const_iterator it = devices_.begin(); it != devices_.end(); ++it)
        prefix += it->name() + it->vendor() + it->driver_version();
      std::string sha1 = tools::sha1(prefix + source);

      std::ifstream cached((cache_path_+sha1).c_str(),std::ios::binary);
      if (cached)
      {
        vcl_size_t len;
        std::vector<char> buffer;

        cached.read((char*)&len, sizeof(vcl_size_t));
        buffer.resize(len);
        cached.read((char*)buffer.data(), std::streamsize(len));
        temp = brig_module(buffer);
      }
    }

    if (temp.empty())
    {
    	//const char * options = build_options_.c_str();
    	compiler_helper helper;
    	std::vector<char> binary = helper.compile_brig(source_text);
    	temp = brig_module(binary);
    }

    programs_.push_back(tools::shared_ptr<hsa::program>(new hsa::program(temp, *this, prog_name)));

    viennacl::hsa::program & prog = *programs_.back();
    //temporary - use single hsa device for tests.
    finalize(current_device().id(), prog);

#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Stored program '" << programs_.back()->name() << "' in context " << h_ << std::endl;
    std::cout << "ViennaCL: There is/are " << programs_.size() << " program(s)" << std::endl;
#endif

    return prog;
  }

  /** @brief Delete the program with the provided name */
  void delete_program(std::string const & name)
  {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Deleting program '" << name << "' from context " << h_ << std::endl;
#endif
    for (program_container_type::iterator it = programs_.begin();
         it != programs_.end();
         ++it)
    {
      if ((*it)->name() == name)
      {
        programs_.erase(it);
        return;
      }
    }
  }

  /** @brief Returns the program with the provided name */
  viennacl::hsa::program & get_program(std::string const & name)
  {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Getting program '" << name << "' from context " << h_ << std::endl;
    std::cout << "ViennaCL: There are " << programs_.size() << " programs" << std::endl;
#endif
    for (program_container_type::iterator it = programs_.begin();
         it != programs_.end();
         ++it)
    {
      //std::cout << "Name: " << (*it)->name() << std::endl;
      if ((*it)->name() == name)
        return **it;
    }
    std::cerr << "ViennaCL: Could not find program '" << name << "'" << std::endl;
    throw "In class 'context': name invalid in get_program()";
    //return programs_[0];  //return a defined object
  }

  viennacl::hsa::program const & get_program(std::string const & name) const
  {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Getting program '" << name << "' from context " << h_ << std::endl;
    std::cout << "ViennaCL: There are " << programs_.size() << " programs" << std::endl;
#endif
    for (program_container_type::const_iterator it = programs_.begin();
         it != programs_.end();
         ++it)
    {
      //std::cout << "Name: " << (*it)->name() << std::endl;
      if ((*it)->name() == name)
        return **it;
    }
    std::cerr << "ViennaCL: Could not find program '" << name << "'" << std::endl;
    throw "In class 'context': name invalid in get_program()";
    //return programs_[0];  //return a defined object
  }

  /** @brief Returns whether the program with the provided name exists or not */
  bool has_program(std::string const & name)
  {
    for (program_container_type::iterator it = programs_.begin();
         it != programs_.end();
         ++it)
    {
      if ((*it)->name() == name) return true;
    }
    return false;
  }

  /** @brief Returns the program with the provided id */
  viennacl::hsa::program & get_program(vcl_size_t id)
  {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Getting program '" << id << "' from context " << h_ << std::endl;
    std::cout << "ViennaCL: There are " << programs_.size() << " programs" << std::endl;
#endif

    if (id >= programs_.size())
      throw invalid_program();

    return *programs_[id];
  }

  program_container_type get_programs()
  {
    return programs_;
  }

  /** @brief Returns the number of programs within this context */
  vcl_size_t program_num() { return programs_.size(); }

  /** @brief Convenience function for retrieving the kernel of a program directly from the context */
  viennacl::hsa::kernel & get_kernel(std::string const & program_name, std::string const & kernel_name) { return get_program(program_name).get_kernel(kernel_name); }

  /** @brief Returns the number of devices within this context */
  vcl_size_t device_num() { return devices_.size(); }

  /** @brief Returns the context handle */
  const viennacl::hsa::handle<hsa_environment> & handle() const { return h_; }

  /** @brief Returns the current build option string */
  std::string build_options() const { return build_options_; }

  /** @brief Sets the build option string, which is passed to the OpenCL compiler in subsequent compilations. Does not effect programs already compiled previously. */
  void build_options(std::string op) { build_options_ = op; }

  /** @brief Less-than comparable for compatibility with std:map  */
  bool operator<(context const & other) const
  {
    return false;
  }

  bool operator==(context const & other) const
  {
    return true;
  }

private:
  /** @brief Initialize a new context. Reuse any previously supplied information (devices, queues) */
  void init_new()
  {
    assert(!initialized_ && bool("ViennaCL FATAL error: Context already created!"));

#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Initializing new ViennaCL context." << std::endl;
#endif

    hsa_environment env;
    env.startup();
    h_ = env;


    std::vector<hsa_agent_t> device_id_array;
    if (devices_.empty()) //get the default device if user has not yet specified a list of devices
    {
      //create an OpenCL context for the provided devices:
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
      std::cout << "ViennaCL: Setting all devices for context..." << std::endl;
#endif


      std::vector<device> devices = h_.get().get_devices(device_type_);
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
      std::cout << "ViennaCL: Number of devices for context: " << devices.size() << std::endl;
#endif
      vcl_size_t device_num = std::min<vcl_size_t>(default_device_num_, devices.size());
      for (vcl_size_t i=0; i<device_num; ++i)
        devices_.push_back(devices[i]);

      if (devices.size() == 0)
      {
        std::cerr << "ViennaCL: FATAL ERROR: No devices of type '";
        switch (device_type_)
        {
        case HSA_DEVICE_TYPE_CPU:          std::cout << "CPU"; break;
        case HSA_DEVICE_TYPE_GPU:          std::cout << "GPU"; break;
        case HSA_DEVICE_TYPE_DSP:  std::cout << "ACCELERATOR"; break;
        default:
          std::cout << "UNKNOWN" << std::endl;
        }
        std::cout << "' found!" << std::endl;
      }
    }

    //extract list of device ids:
    for (std::vector< viennacl::hsa::device >::const_iterator iter = devices_.begin();
         iter != devices_.end();
         ++iter)
      device_id_array.push_back(iter->id());
    initialized_ = true;
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Initialization of new ViennaCL context done." << std::endl;
#endif
  }

  /** @brief Reuses a supplied context. */
  void init_existing(hsa_environment c)
  {
    assert(!initialized_ && bool("ViennaCL FATAL error: Context already created!"));
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Initialization of ViennaCL context from existing context." << std::endl;
#endif

    //set context handle:
    h_ = c;
    h_.inc(); // if the user provides the context, then the user will also call release() on the context. Without inc(), we would get a seg-fault due to double-free at program termination.
    std::vector<hsa_agent_t> device_id_array;
    if (devices_.empty()) //get the default device if user has not yet specified a list of devices
    {
      //create an OpenCL context for the provided devices:
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
      std::cout << "ViennaCL: Setting all devices for context..." << std::endl;
#endif


      std::vector<device> devices = h_.get().get_devices(device_type_);
 #if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
       std::cout << "ViennaCL: Number of devices for context: " << devices.size() << std::endl;
 #endif
       vcl_size_t device_num = std::min<vcl_size_t>(default_device_num_, devices.size());
       for (vcl_size_t i=0; i<device_num; ++i)
         devices_.push_back(devices[i]);

       if (devices.size() == 0)
       {
         std::cerr << "ViennaCL: FATAL ERROR: No devices of type '";
         switch (device_type_)
         {
         case HSA_DEVICE_TYPE_CPU:          std::cout << "CPU"; break;
         case HSA_DEVICE_TYPE_GPU:          std::cout << "GPU"; break;
         case HSA_DEVICE_TYPE_DSP:  std::cout << "ACCELERATOR"; break;
         default:
           std::cout << "UNKNOWN" << std::endl;
         }
         std::cout << "' found!" << std::endl;
       }
     }

    //extract list of device ids:
    for (std::vector< viennacl::hsa::device >::const_iterator iter = devices_.begin();
         iter != devices_.end();
         ++iter)
      device_id_array.push_back(iter->id());
    initialized_ = true;
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
    std::cout << "ViennaCL: Initialization of new ViennaCL context done." << std::endl;
#endif
  }

  hsa_status_t finalize(hsa_agent_t device, program& p)
  {
	  hsa_status_t status = HSA_STATUS_SUCCESS;
      hsa_ext_program_handle_t hsa_program;
      hsa_program.handle = 0;
	    //Create hsa program.
	    status = hsa_ext_program_create(&device, 1, HSA_EXT_BRIG_MACHINE_LARGE, HSA_EXT_BRIG_PROFILE_FULL, &hsa_program);

	    //Add BRIG module to hsa program.
	    hsa_ext_brig_module_handle_t module;
	    status = hsa_ext_add_module(hsa_program, p.handle().get().brig_module_, &module);
	    // entry offset into the code section.
	    std::vector<hsa_ext_finalization_request_t> finalization_request_list;
	    std::for_each(p.handle().get().kernels_.begin(), p.handle().get().kernels_.end(), [&finalization_request_list, &module](const kernel_entry& entry){
	    	hsa_ext_finalization_request_t request;
	    	memset(&request, 0, sizeof(hsa_ext_finalization_request_t));
	    	request.module = module;              // module handle.
	    	request.program_call_convention = 0;  // program call convention. not supported.
	    	request.symbol = entry.offset_;
	    	finalization_request_list.push_back(request);
	    } );

	    //Finalize hsa program.
	    status = hsa_ext_finalize_program(hsa_program, device, finalization_request_list.size(), &finalization_request_list[0], NULL, NULL, 0, NULL, 0);


	    hsa_region_t region;
    	hsa_agent_iterate_regions(device,context::get_kernarg, &region);

	    for (size_t i = 0 ;i < finalization_request_list.size() ; ++i)
	    {
	    	// create kernargs
		    hsa_ext_code_descriptor_t *hsa_code_descriptor;
		    hsa_ext_query_kernel_descriptor_address(hsa_program, module, finalization_request_list[i].symbol, &hsa_code_descriptor);
	    	size_t run_kernel_arg_buffer_size =  hsa_code_descriptor->kernarg_segment_byte_size;
	    	viennacl::hsa::kernel_arg_buffer buffer(region, run_kernel_arg_buffer_size,  p.handle().get().kernels_[i].arg_count_ );
	    	p.add_kernel(tools::shared_ptr<viennacl::hsa::kernel>(new kernel(hsa_code_descriptor, buffer,p,*this, p.handle().get().kernels_[i].name_ )));
	    }
	    return status;
  }


private:
 static hsa_status_t get_kernarg(hsa_region_t region, void* data) {
    hsa_region_flag_t flags;
    hsa_region_get_info(region, HSA_REGION_INFO_FLAGS, &flags);
    if (flags & HSA_REGION_FLAG_KERNARG) {
      hsa_region_t * ret = (hsa_region_t *) data;
      *ret = region;
      return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
  }



  bool initialized_;
  std::string cache_path_;
  hsa_device_type_t device_type_;
  viennacl::hsa::handle<hsa_environment> h_;

  std::vector< viennacl::hsa::device > devices_;
  vcl_size_t current_device_id_;
  vcl_size_t default_device_num_;
  program_container_type programs_;
  std::map< hsa_agent_t, std::vector< viennacl::hsa::command_queue> > queues_;
  std::string build_options_;
  vcl_size_t pf_index_;
  vcl_size_t current_queue_id_;
}; //context



/** @brief Returns the kernel with the provided name */
inline viennacl::hsa::kernel & viennacl::hsa::program::get_kernel(std::string const & name)
{
  //std::cout << "Requiring kernel " << name << " from program " << name_ << std::endl;
  for (kernel_container_type::iterator it = kernels_.begin();
       it != kernels_.end();
       ++it)
  {
    if (((*it)->name() == name) || ((*it)->name() == "&__OpenCL_"+name+"_kernel"))
      return **it;
  }
  std::cerr << "ViennaCL: FATAL ERROR: Could not find kernel '" << name << "' from program '" << name_ << "'" << std::endl;
  std::cout << "Number of kernels in program: " << kernels_.size() << std::endl;
  throw "Kernel not found";
  //return kernels_[0];  //return a defined object
}


inline void viennacl::hsa::kernel::set_work_size_defaults()
{
  assert( p_program_ != NULL && bool("Kernel not initialized, program pointer invalid."));
  assert( p_context_ != NULL && bool("Kernel not initialized, context pointer invalid."));
  local_work_size_[0] = 256;      local_work_size_[1] = 0;  local_work_size_[2] = 0;
  global_work_size_[0] = 256*128; global_work_size_[1] = 0; global_work_size_[2] = 0;
}

}
}

#endif
