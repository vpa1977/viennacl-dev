/*
 * brig_helper.hpp
 *
 *  Created on: 19/03/2015
 *      Author: bsp
 */

#ifndef VIENNACL_HSA_BRIG_HELPER_HPP_
#define VIENNACL_HSA_BRIG_HELPER_HPP_

#include <hsa.h>
#include <hsa_ext_finalize.h>
#include <hsa_ext_image.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <vector>
#include <stdexcept>
/*

namespace viennacl {
namespace hsa {

struct kernel_entry
{
	size_t offset_;
	size_t arg_count_;
	std::string name_;
};

class brig_module {
	enum brig_module_status {
		STATUS_SUCCESS = 0,
		STATUS_KERNEL_INVALID_SECTION_HEADER = 1,
		STATUS_KERNEL_ELF_INITIALIZATION_FAILED = 2,
		STATUS_KERNEL_INVALID_ELF_CONTAINER = 3,
		STATUS_KERNEL_MISSING_DATA_SECTION = 4,
		STATUS_KERNEL_MISSING_CODE_SECTION = 5,
		STATUS_KERNEL_MISSING_OPERAND_SECTION = 6,
		STATUS_UNKNOWN = 7,
	};

	enum brig_kinds {
	    BRIG_KIND_NONE = 0x0000,
	    BRIG_KIND_DIRECTIVE_BEGIN = 0x1000,
	    BRIG_KIND_DIRECTIVE_KERNEL = 0x1008,
	};


	enum {
		SECTION_HSA_DATA = 0, SECTION_HSA_CODE, SECTION_HSA_OPERAND,
	};

	struct section_desc {
		int section_id;
		const char *brig_name;
		const char *bif_name;
	};

	struct brig_base {
	    uint16_t byte_count;
	    uint16_t kind;
	};

	struct brig_directive_executable {
	    uint16_t byteCount;
	    uint16_t kind;
	    uint32_t name;
	    uint16_t outArgCount;
	    uint16_t inArgCount;
	    uint32_t firstInArg;
	    uint32_t firstCodeBlockEntry;
	    uint32_t nextModuleEntry;
	    uint32_t codeBlockEntryCount;
	    uint8_t modifier;
	    uint8_t linkage;
	    uint16_t reserved;
	};

	struct brig_data {
	    uint32_t byteCount;
	    uint8_t bytes[1];
	};


public:
	brig_module() {}
	brig_module(const std::vector<char>& data) {
		brig_module_ = (hsa_ext_module_t) malloc(data.size());
		memcpy(brig_module_, &data[0], data.size());
	}
	bool empty() { return kernels_.size() == 0; }
	void free_module()
	{
		 free (brig_module_);
	}

	std::vector<kernel_entry> kernels_;
	hsa_ext_module_t brig_module_;



};

}
}
*/

#endif /* VIENNACL_HSA_BRIG_HELPER_HPP_ */
