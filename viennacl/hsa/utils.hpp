#ifndef VIENNACL_HSA_UTILS_HPP_
#define VIENNACL_HSA_UTILS_HPP_

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

/** @file viennacl/hsa/utils.hpp
    @brief Provides HSA-related utilities.
 */

#include <vector>
#include <string>
#include "viennacl/hsa/backend.hpp"
#include "viennacl/hsa/device.hpp"

namespace viennacl
{
  namespace ocl
  {

    template<>
    struct DOUBLE_PRECISION_CHECKER<double, viennacl::hsa::context>
    {

      static void apply(viennacl::hsa::context const & ctx)
      {
        if (!ctx.current_device().double_support())
          throw viennacl::ocl::double_precision_not_provided_error();
      }
    };
  }
}

#endif
