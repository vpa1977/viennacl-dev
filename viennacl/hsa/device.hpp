#ifndef VIENNACL_HSA_DEVICE_HPP_
#define VIENNACL_HSA_DEVICE_HPP_

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

/** @file viennacl/hsa/device.hpp
    @brief Represents a HSA device within ViennaCL
 */


#include <hsa.h>
#include <hsa_ext_finalize.h>
#include <hsa_ext_image.h>

#include<stdio.h>

#include <vector>
#include <string>
#include <sstream>
#include <assert.h>
#include "viennacl/ocl/device_utils.hpp" // including ocl type because it will require changing all maps
#include "viennacl/hsa/handle.hpp"
#include "viennacl/hsa/error.hpp"
#include "viennacl/device.hpp"

namespace viennacl
{
  namespace hsa
  {

    /** @brief A class representing a compute device (e.g. a GPU)
     *  @TODO: update device definition to query info properly 
     */
    class device : public viennacl::device_capabilities
    {
      std::string NOT_IMPLEMENTED;
    public:

      explicit device() : device_()
      {
      }

      explicit device(hsa_agent_t dev) : device_(dev)
      {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_DEVICE)
        std::cout << "ViennaCL: Creating device object (CTOR with hsa_agent_t)" << std::endl;
#endif
      }

      device(const device & other) : device_()
      {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_DEVICE)
        std::cout << "ViennaCL: Creating device object (Copy CTOR)" << std::endl;
#endif
        if (device_.handle != other.device_.handle)
        {
          device_ = other.device_;
        }
      }

      const hsa_agent_t& id() const
      {
        return device_;
      }

      const std::string name() const
      {
        return name_;
      }

      const std::string driver_version() const
      {
        return NOT_IMPLEMENTED;
      }

      const std::string vendor() const
      {
        return NOT_IMPLEMENTED;
      }

      bool double_support() const
      {
        return true;
      }

      size_t max_compute_units() const
      {
        return 6; // Kaveri
      }

      const std::string double_support_extension() const
      {
        return "cl_khr_fp64";
      }

      bool operator==(device const & other) const
      {
        return device_.handle == other.device_.handle;
      }

      bool operator==(hsa_agent_t other) const
      {
        return device_.handle == other.handle;
      }

      size_t local_mem_size() const
      {
        return 32768;
      }

      size_t max_work_group_size() const
      {
        return 256;
      }

      std::vector<vcl_size_t> max_work_item_sizes() const
      {
        std::vector<vcl_size_t> ret;
        ret.push_back(256);
        ret.push_back(256);
        ret.push_back(256);
        return ret;
      }

      int vendor_id() const
      {
        return 4098;
      }

      int type() const
      {
        return CL_DEVICE_TYPE_GPU;
      }

      viennacl::ocl::device_architecture_family architecture_family() const
      {
        return viennacl::ocl::northern_islands;
      }
    private:

      void init()
      {
        // read hsa agent id
        NOT_IMPLEMENTED = "Not implemented";
        name_ = "Kaveri"; // temporary
      }


      hsa_agent_t device_;
      std::string id_;
      std::string name_;

    };

  } //namespace hsa
} //namespace viennacl

#endif
