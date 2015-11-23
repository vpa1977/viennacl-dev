#ifndef VIENNACL_DEVICE_CAPABILITIES_HPP_
#define VIENNACL_DEVICE_CAPABILITIES_HPP_

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
    @brief Abstract device class.
 */


#include "CL/cl.h"
#include<stdio.h>
#include <vector>
#include <string>
#include <sstream>
#include <assert.h>
#include "viennacl/ocl/device_utils.hpp"


namespace viennacl
{
  

  /** @brief A class representing a compute device device_capabilities (e.g. a GPU)
   */
  class device_capabilities
  {
  public:
    virtual const std::string name() const = 0;
    virtual const std::string driver_version() const = 0;
    virtual const std::string vendor() const = 0;
    virtual bool double_support() const = 0;
    virtual size_t max_compute_units() const = 0;
    virtual const std::string double_support_extension() const = 0;
		virtual cl_ulong local_mem_size() const = 0;
    virtual size_t max_work_group_size() const = 0;
    virtual std::vector<vcl_size_t> max_work_item_sizes() const = 0;
    virtual int vendor_id() const = 0;
    virtual int type() const = 0;
    virtual viennacl::ocl::device_architecture_family architecture_family() const = 0;
  };



} //namespace viennacl

#endif
