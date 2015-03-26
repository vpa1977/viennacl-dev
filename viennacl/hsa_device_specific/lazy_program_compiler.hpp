#ifndef VIENNACL_HSA_DEVICE_SPECIFIC_LAZY_PROGRAM_COMPILER_HPP
#define VIENNACL_HSA_DEVICE_SPECIFIC_LAZY_PROGRAM_COMPILER_HPP

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


/** @file viennacl/hsa_device_specific/lazy_program_compiler.hpp
    @brief Helper for compiling a program lazily
*/

#include <map>

#include "viennacl/ocl/context.hpp"
#include "viennacl/hsa/context.hpp"

namespace viennacl
{

namespace hsa_specific
{

#ifdef VIENNACL_WITH_HSA
namespace vcl = viennacl::hsa;
#else
namespace vcl = viennacl::ocl;
#endif

  class lazy_program_compiler
  {
  public:

    lazy_program_compiler(vcl::context * ctx, std::string const & name, std::string const & src, bool force_recompilation) : ctx_(ctx), name_(name), src_(src), force_recompilation_(force_recompilation){ }
    lazy_program_compiler(vcl::context * ctx, std::string const & name, bool force_recompilation) : ctx_(ctx), name_(name), force_recompilation_(force_recompilation){ }

    void add(std::string const & src) {  src_+=src; }

    std::string const & src() const { return src_; }

    vcl::program & program()
    {
      if (force_recompilation_ && ctx_->has_program(name_))
        ctx_->delete_program(name_);
      if (!ctx_->has_program(name_))
      {
#ifdef VIENNACL_BUILD_INFO
          std::cerr << "Creating program " << program_name << std::endl;
#endif
          ctx_->add_program(src_, name_);
#ifdef VIENNACL_BUILD_INFO
          std::cerr << "Done creating program " << program_name << std::endl;
#endif
      }
      return ctx_->get_program(name_);
    }

  private:
    vcl::context * ctx_;
    std::string name_;
    std::string src_;
    bool force_recompilation_;
  };

}

}
#endif
