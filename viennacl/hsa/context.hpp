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
#include "viennacl/tools/sha1.hpp"
#include "viennacl/tools/shared_ptr.hpp"

namespace viennacl
{
  namespace hsa
  {

    /** @brief Manages an HSA context and provides the respective convenience functions for creating buffers, etc.
     *
     * This class was originally written before the OpenCL C++ bindings were standardized.
     * Regardless, it provides a couple of convience functionality which is not covered by the OpenCL C++ bindings.
     */
    class context : public viennacl::program_compiler
    {
      typedef std::vector< tools::shared_ptr<viennacl::hsa::program> > program_container_type;

public:
  typedef device device_type;

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
      std::string cache_path() const
      {
        return cache_path_;
      }

      /** @brief Sets the compiled kernel cache path */
      void cache_path(std::string new_path)
      {
        cache_path_ = new_path;
      }

      //////// Get and set default number of devices per context */

      /** @brief Returns the maximum number of devices to be set up for the context */
      vcl_size_t default_device_num() const
      {
        return default_device_num_;
      }

      /** @brief Sets the maximum number of devices to be set up for the context */
      void default_device_num(vcl_size_t new_num)
      {
        default_device_num_ = new_num;
      }

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
        std::cout << "ViennaCL: Setting new device type for context " << dtype << std::endl;
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
        std::cout << "ViennaCL: Setting new current device for context " << d.name() << std::endl;
#endif
        bool found = false;
        for (vcl_size_t i = 0; i < devices_.size(); ++i)
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
        std::cout << "ViennaCL: Adding new device to context " << d.name() << std::endl;
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
      void init(hsa_environment* c)
      {
        init_existing(c);
      }


      //////////////////// create queues ////////////////////////////////

      /** @brief Adds an existing queue for the given device to the context */
      void add_queue(hsa_agent_t dev, hsa_queue_t* q)
      {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
        std::cout << "ViennaCL: Adding existing queue " << q << " for device " << dev << " to context " << h_ << std::endl;
#endif
        viennacl::hsa::handle<hsa_queue_t*> queue_handle(q, *this);
        queues_[dev.handle].push_back(viennacl::hsa::command_queue(queue_handle));
        queues_[dev.handle].back().handle().inc();
      }

      void add_queue(uint64_t device_handle, hsa_queue_t* q)
      {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
        std::cout << "ViennaCL: Adding existing queue " << q << " for device " << dev << " to context " << h_ << std::endl;
#endif
        viennacl::hsa::handle<hsa_queue_t*> queue_handle(q, *this);
        queues_[device_handle].push_back(viennacl::hsa::command_queue(queue_handle));
        queues_[device_handle].back().handle().inc();
      }

      /** @brief Adds a queue for the given device to the context */
      void add_queue(hsa_agent_t dev)
      {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
        std::cout << "ViennaCL: Adding new queue for device " << dev << " to context " << h_ << std::endl;
#endif
        uint32_t queue_size = 0;
        hsa_status_t status = hsa_agent_get_info(dev, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
        if (status != HSA_STATUS_SUCCESS)
          throw std::runtime_error("unable to get queue size");
        hsa_queue_t* command_queue;
        status = hsa_queue_create(dev, queue_size, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX, UINT32_MAX, &command_queue);
        if (status != HSA_STATUS_SUCCESS)
                  throw std::runtime_error("unable to create HSA queue");

        viennacl::hsa::command_queue queue(viennacl::hsa::handle<hsa_queue_t*>(command_queue, *this));
        queues_[dev.handle].push_back(queue);
      }

      /** @brief Adds a queue for the given device to the context */
      void add_queue(viennacl::hsa::device d)
      {
        add_queue(d.id());
      }

      //get queue for default device:

      viennacl::hsa::command_queue & get_queue()
      {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
        std::cout << "ViennaCL: Getting queue for device " << devices_[current_device_id_].name() << " in context " << h_ << std::endl;
        std::cout << "ViennaCL: Current queue id " << current_queue_id_ << std::endl;
#endif

        return queues_[devices_[current_device_id_].id().handle][current_queue_id_];
      }

      viennacl::hsa::command_queue const & get_queue() const
      {
        typedef std::map< uint64_t, std::vector<viennacl::hsa::command_queue> > QueueContainer;

#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
        std::cout << "ViennaCL: Getting const queue for device " << devices_[current_device_id_].name() << " in context " << h_ << std::endl;
        std::cout << "ViennaCL: Current queue id " << current_queue_id_ << std::endl;
#endif

        // find queue:
        QueueContainer::const_iterator it = queues_.find(devices_[current_device_id_].id().handle);
        if (it != queues_.end())
        {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
          std::cout << "ViennaCL: Queue handle " << (it->second)[current_queue_id_].handle() << std::endl;
#endif
          return (it->second)[current_queue_id_];
        }

        std::cerr << "ViennaCL: FATAL ERROR: Could not obtain current command queue!" << std::endl;
        std::cout << "Number of queues in context: " << queues_.size() << std::endl;
        std::cout << "Number of devices in context: " << devices_.size() << std::endl;
        throw "queue not found!";
      }

      //get a particular queue:

      /** @brief Returns the queue with the provided index for the given device */
      viennacl::hsa::command_queue & get_queue(hsa_agent_t dev, vcl_size_t i = 0)
      {
        if (i >= queues_[dev.handle].size())
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

        return queues_[devices_[device_index].id().handle][i];
      }

      /** @brief Returns the current device */
      // TODO: work out the const issues

      viennacl::hsa::command_queue const & current_queue() //const
      {
        return queues_[devices_[current_device_id_].id().handle][current_queue_id_];
      }

      /** @brief Switches the current device to the i-th device in this context */
      void switch_queue(vcl_size_t i)
      {
        assert(i < queues_[devices_[current_device_id_].id().handle].size() && bool("In class 'context': Provided queue index out of range for device!"));
        current_queue_id_ = i;
      }

      /** @brief If the supplied command_queue is used within the context, it becomes the current active command_queue, the command_queue's device becomes current active device. */
      void switch_queue(viennacl::hsa::command_queue const & q)
      {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
        std::cout << "ViennaCL: Setting new current queue for context " << h_ << std::endl;
#endif
        bool found = false;
        typedef std::map< uint64_t, std::vector<viennacl::hsa::command_queue> > QueueContainer;

        // For each device:
        vcl_size_t j = 0;
        for (QueueContainer::const_iterator it = queues_.begin(); it != queues_.end(); it++, j++)
        {
          const std::vector<viennacl::hsa::command_queue> & qv = (it->second);
          // For each queue candidate
          for (vcl_size_t i = 0; i < qv.size(); ++i)
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
      viennacl::hsa::program & add_program(const std::vector<char>& binary, std::string const & prog_name)
      {
        programs_.push_back(tools::shared_ptr<hsa::program>(new viennacl::hsa::program(binary, *this, prog_name)));
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
        std::cout << "ViennaCL: Adding program '" << prog_name << "' with cl_program to context " << h_ << std::endl;
#endif
        return *programs_.back();
      }
      
      void compile_program(std::string const & source, std::string const & prog_name)
      {
        add_program(source, prog_name);
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

        std::vector<char> temp;

        //
        // Retrieves the program in the cache
        //
        if (cache_path_.size())
        {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
          std::cout << "ViennaCL: Cache at " << cache_path_ << std::endl;
#endif

          std::string prefix;
          for (std::vector< viennacl::hsa::device >::const_iterator it = devices_.begin(); it != devices_.end(); ++it)
            prefix += it->name() + it->vendor() + it->driver_version();
          std::string sha1 = tools::sha1(prefix + source);

          std::ifstream cached((cache_path_ + sha1).c_str(), std::ios::binary);
          if (cached)
          {
            vcl_size_t len;
            std::vector<char> buffer;

            cached.read((char*) &len, sizeof (vcl_size_t));
            buffer.resize(len);
            cached.read((char*) buffer.data(), std::streamsize(len));
            temp = buffer;
          }
        }

        if (temp.empty())
        {
          //const char * options = build_options_.c_str();
          compiler_helper helper;
          std::vector<char> binary = helper.compile_brig(source_text);
          temp = binary;
        }

        programs_.push_back(tools::shared_ptr<hsa::program>(new hsa::program(temp, *this, prog_name)));

        viennacl::hsa::program & prog = *programs_.back();
        //temporary - use single hsa device for tests.
        //finalize(current_device().id(), prog);

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
      viennacl::abstract_program & get_program(std::string const & name)
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

      viennacl::abstract_program const & get_program(std::string const & name) const
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
      vcl_size_t program_num()
      {
        return programs_.size();
      }

      /** @brief Convenience function for retrieving the kernel of a program directly from the context */
      viennacl::hsa::kernel & get_kernel(std::string const & program_name, std::string const & kernel_name)
      {
        return (viennacl::hsa::kernel &) get_program(program_name).kernel(kernel_name);
      }

      /** @brief Returns the number of devices within this context */
      vcl_size_t device_num()
      {
        return devices_.size();
      }

      /** @brief Returns the context handle */
      const viennacl::hsa::handle<hsa_environment*> & handle() const
      {
        return h_;
      }

      /** @brief Returns the current build option string */
      std::string build_options() const
      {
        return build_options_;
      }

      /** @brief Sets the build option string, which is passed to the OpenCL compiler in subsequent compilations. Does not effect programs already compiled previously. */
      void build_options(std::string op)
      {
        build_options_ = op;
      }

      /** @brief Less-than comparable for compatibility with std:map  */
      bool operator<(context const & other) const
      {
        return h_.get() < other.handle().get();
      }

      bool operator==(context const & other) const
      {
        return h_.get() == other.handle().get();
      }

    private:

      /** @brief Initialize a new context. Reuse any previously supplied information (devices, queues) */
      void init_new()
      {
        assert(!initialized_ && bool("ViennaCL FATAL error: Context already created!"));

#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
        std::cout << "ViennaCL: Initializing new ViennaCL context." << std::endl;
#endif

        hsa_environment* env = new hsa_environment();
        env->startup();
        h_ = env;


        std::vector<hsa_agent_t> device_id_array;
        if (devices_.empty()) //get the default device if user has not yet specified a list of devices
        {
          //create an OpenCL context for the provided devices:
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
          std::cout << "ViennaCL: Setting all devices for context..." << std::endl;
#endif


          std::vector<device> devices = h_.get()->get_devices(device_type_);
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
          std::cout << "ViennaCL: Number of devices for context: " << devices.size() << std::endl;
#endif
          vcl_size_t device_num = std::min<vcl_size_t>(default_device_num_, devices.size());
          for (vcl_size_t i = 0; i < device_num; ++i)
            devices_.push_back(devices[i]);

          if (devices.size() == 0)
          {
            std::cerr << "ViennaCL: FATAL ERROR: No devices of type '";
            switch (device_type_)
            {
              case HSA_DEVICE_TYPE_CPU: std::cout << "CPU";
                break;
              case HSA_DEVICE_TYPE_GPU: std::cout << "GPU";
                break;
              case HSA_DEVICE_TYPE_DSP: std::cout << "ACCELERATOR";
                break;
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
      void init_existing(hsa_environment* c)
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


          std::vector<device> devices = h_.get()->get_devices(device_type_);
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_CONTEXT)
          std::cout << "ViennaCL: Number of devices for context: " << devices.size() << std::endl;
#endif
          vcl_size_t device_num = std::min<vcl_size_t>(default_device_num_, devices.size());
          for (vcl_size_t i = 0; i < device_num; ++i)
            devices_.push_back(devices[i]);

          if (devices.size() == 0)
          {
            std::cerr << "ViennaCL: FATAL ERROR: No devices of type '";
            switch (device_type_)
            {
              case HSA_DEVICE_TYPE_CPU: std::cout << "CPU";
                break;
              case HSA_DEVICE_TYPE_GPU: std::cout << "GPU";
                break;
              case HSA_DEVICE_TYPE_DSP: std::cout << "ACCELERATOR";
                break;
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



    private:


      bool initialized_;
      std::string cache_path_;
      hsa_device_type_t device_type_;
      viennacl::hsa::handle<hsa_environment*> h_;

      std::vector< viennacl::hsa::device > devices_;
      vcl_size_t current_device_id_;
      vcl_size_t default_device_num_;
      program_container_type programs_;
      std::map< uint64_t, std::vector< viennacl::hsa::command_queue> > queues_;
      std::string build_options_;
      vcl_size_t pf_index_;
      vcl_size_t current_queue_id_;
    }; //context

    /** @brief Returns the kernel with the provided name */
    inline viennacl::hsa::kernel & viennacl::hsa::program::get_kernel(std::string const & name)
    {
      std::string candidate = "&__OpenCL_" + name + "_kernel";
      //std::cout << "Requiring kernel " << name << " from program " << name_ << std::endl;
      for (kernel_container_type::iterator it = kernels_.begin();
              it != kernels_.end();
              ++it)
      {
        std::string local_name = (*it)->name();
        if (!strcmp(local_name.c_str(), name.c_str()) || !strcmp(local_name.c_str(), candidate.c_str()))
          return **it;
      }
      std::cerr << "ViennaCL: FATAL ERROR: Could not find kernel '" << name << "' from program '" << name_ << "'" << std::endl;
      std::cout << "Number of kernels in program: " << kernels_.size() << std::endl;
      for (kernel_container_type::iterator it = kernels_.begin();
              it != kernels_.end();
              ++it)
      {
        std::cout << "Candidate: " << (*it)->name() << std::endl;
      }
      throw "Kernel not found";
    }

    inline void viennacl::hsa::program::finalize()
    {
      hsa_agent_t agent = p_context_->current_device().id();
      hsa_status_t err;
      hsa_ext_program_t program;
      memset(&program, 0, sizeof (hsa_ext_program_t));
      err = hsa_ext_program_create(HSA_MACHINE_MODEL_LARGE, HSA_PROFILE_FULL,
              HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, NULL, &program);
      /*
       * Add the BRIG module to hsa program.
       */
      err = hsa_ext_program_add_module(program, (hsa_ext_module_t) & source_[0]);

      hsa_isa_t isa;
      err = hsa_agent_get_info(p_context_->current_device().id(),
              HSA_AGENT_INFO_ISA, &isa);
      /*
       * Finalize the program and extract the code object.
       */
      hsa_ext_control_directives_t control_directives;
      memset(&control_directives, 0, sizeof (hsa_ext_control_directives_t));
      hsa_code_object_t code_object;
      err = hsa_ext_program_finalize(program, isa, 0, control_directives, "",
              HSA_CODE_OBJECT_TYPE_PROGRAM, &code_object);

      err = hsa_ext_program_destroy(program);

      hsa_executable_t executable;
      err = hsa_executable_create(HSA_PROFILE_FULL,
              HSA_EXECUTABLE_STATE_UNFROZEN, "", &executable);

      /*
       * Load the code object.
       */
      err = hsa_executable_load_code_object(executable, agent, code_object,
              "");

      /*
       * Freeze the executable; it can now be queried for symbols.
       */
      err = hsa_executable_freeze(executable, "");
      if (err == HSA_STATUS_SUCCESS)
      {
        hsa_code new_code;
        new_code.code_object_ = code_object;
        new_code.executable_ = executable;

        handle_ = viennacl::hsa::handle<hsa_code>(new_code, *p_context_);


        load_kernels();
      }
    }

    inline void viennacl::hsa::program::load_kernels()
    {

      auto get_kernarg = [](hsa_region_t region, void* data)->hsa_status_t {
        hsa_region_segment_t segment;
        hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
        if (HSA_REGION_SEGMENT_GLOBAL != segment)
        {
          return HSA_STATUS_SUCCESS;
        }

        hsa_region_global_flag_t flags;
        hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
        if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG)
        {
          hsa_region_t* ret = (hsa_region_t*) data;
          *ret = region;
          return HSA_STATUS_INFO_BREAK;
        }

        return HSA_STATUS_SUCCESS;
      };


      kernarg_region_.handle = (uint64_t) - 1;
      hsa_agent_iterate_regions(p_context()->current_device().id(), get_kernarg, &kernarg_region_);


      hsa_executable_iterate_symbols(handle_.get().executable_, [](hsa_executable_t /*executable*/, hsa_executable_symbol_t symbol, void* data)->hsa_status_t {
        viennacl::hsa::program& prg = *(viennacl::hsa::program*)data;
        hsa_symbol_kind_t kind;
                hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind);
        if (kind == HSA_SYMBOL_KIND_KERNEL)
        {
          hsa_status_t err;
                  uint32_t len;
                  std::string kernel_name;
                  uint64_t kernel_object;
                  uint32_t kernarg_segment_size;
                  uint32_t group_segment_size;
                  uint32_t private_segment_size;

                  // create a new kernel
                  err = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &len);
                  kernel_name.resize(len);
                  err = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, &kernel_name[0]);

                  err = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object);

                  err = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kernarg_segment_size);

                  err = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &group_segment_size);

                  err = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &private_segment_size);

                  assert(err == HSA_STATUS_SUCCESS);

                  viennacl::hsa::kernel_arg_buffer kernargs(prg.kernarg_region_, kernarg_segment_size);
                  tools::shared_ptr<viennacl::hsa::kernel> kern_ptr(new viennacl::hsa::kernel(kernel_object, kernargs, prg, *prg.p_context(), kernel_name, group_segment_size, private_segment_size));
                  prg.add_kernel(kern_ptr);

        }

        return HSA_STATUS_SUCCESS;
      }, this);
    }

    inline void viennacl::hsa::kernel::set_work_size_defaults()
    {
      assert(p_program_ != NULL && bool("Kernel not initialized, program pointer invalid."));
      assert(p_context_ != NULL && bool("Kernel not initialized, context pointer invalid."));
      local_work_size_[0] = 256;
      local_work_size_[1] = 0;
      local_work_size_[2] = 0;
      global_work_size_[0] = 256 * 128;
      global_work_size_[1] = 0;
      global_work_size_[2] = 0;
    }
    
    inline void viennacl::hsa::kernel::enqueue()
    {
      const viennacl::hsa::command_queue& queue = p_context_->get_queue();
      viennacl::hsa::enqueue(*this, queue);
    }

  }
}

#endif
