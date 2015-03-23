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
#include <libelf.h>
#include <string.h>
#include <stdio.h>
#include <vector>
#include <stdexcept>

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
		uint32_t number_of_sections = 3;

		brig_module_ = (hsa_ext_brig_module_t*) (malloc(
				sizeof(hsa_ext_brig_module_t)
						+ sizeof(void*) * number_of_sections));
		brig_module_->section_count = number_of_sections;

		Elf32_Ehdr* ehdr = NULL;
		Elf_Data *secHdr = NULL;
		Elf_Scn* scn = NULL;
		brig_module_status status;

		if (elf_version( EV_CURRENT) == EV_NONE)
			throw std::runtime_error("elf kernel initialization failed");

		Elf* p_elf = elf_memory( const_cast<char*>(reinterpret_cast<const char*>(&data[0])), data.size());
		if (elf_kind(p_elf) != ELF_K_ELF)
			throw std::runtime_error("invalid elf container");

		if (((ehdr = elf32_getehdr(p_elf)) == NULL)
				|| ((scn = elf_getscn(p_elf, ehdr->e_shstrndx)) == NULL)
				|| ((secHdr = elf_getdata(scn, NULL)) == NULL)) {
			throw std::runtime_error("invalid section header");
		}

		status = extract_section_and_copy(p_elf, secHdr,
				get_section_desc(SECTION_HSA_DATA), brig_module_,
				HSA_EXT_BRIG_SECTION_DATA);

		if (status != STATUS_SUCCESS) {
			throw std::runtime_error("missing data section");
		}

		status = extract_section_and_copy(p_elf, secHdr,
				get_section_desc(SECTION_HSA_CODE), brig_module_,
				HSA_EXT_BRIG_SECTION_CODE);

		if (status != STATUS_SUCCESS) {
			throw std::runtime_error("missing code section");
		}

		status = extract_section_and_copy(p_elf, secHdr,
				get_section_desc(SECTION_HSA_OPERAND), brig_module_,
				HSA_EXT_BRIG_SECTION_OPERAND);

		if (status != STATUS_SUCCESS) {
			throw std::runtime_error("missing operand section");
		}
		elf_end (p_elf);

		list_kernels();
	}
	bool empty() { return kernels_.size() == 0; }
	void free_module()
	{
		 for (size_t i=0; i<brig_module_->section_count; i++) {
		        free (brig_module_->section[i]);
		    }
		    free (brig_module_);
	}

	std::vector<kernel_entry> kernels_;
	hsa_ext_brig_module_t* brig_module_;
private:
	/* list kernel names and offsets &*/
	void list_kernels()
	{
		  /*
		     * Get the data section
		     */
		    hsa_ext_brig_section_header_t* data_section_header =
		                brig_module_->section[HSA_EXT_BRIG_SECTION_DATA];
		    /*
		     * Get the code section
		     */
		    hsa_ext_brig_section_header_t* code_section_header =
		             brig_module_->section[HSA_EXT_BRIG_SECTION_CODE];

		    /*
		     * First entry into the BRIG code section
		     */
		    uint32_t code_offset = code_section_header->header_byte_count;
		    brig_base* code_entry = (brig_base*) ((char*)code_section_header + code_offset);
		    while (code_offset != code_section_header->byte_count) {
		        if (code_entry->kind == BRIG_KIND_DIRECTIVE_KERNEL) {
		            /*
		             * Now find the data in the data section
		             */
		        	brig_directive_executable* directive_kernel = (brig_directive_executable*) (code_entry);
		            uint32_t data_name_offset = directive_kernel->name;
		            brig_data* data_entry = (brig_data*)((char*) data_section_header + data_name_offset);
		            kernel_entry kernel;
		            kernel.name_ = std::string((char*)data_entry->bytes, data_entry->byteCount);
		            kernel.offset_ = code_offset;
		            kernel.arg_count_  = directive_kernel->inArgCount;
		            kernels_.push_back(kernel);
		        }
		        code_offset += code_entry->byte_count;
		        code_entry = (brig_base*) ((char*)code_section_header + code_offset);
		    }

	}

	/* Extract section and copy into HsaBrig */
	brig_module_status extract_section_and_copy(Elf *elfP, Elf_Data *secHdr,
			const section_desc* desc, hsa_ext_brig_module_t* brig_module,
			hsa_ext_brig_section_id_t section_id) {
		Elf_Scn* scn = NULL;
		Elf_Data* data = NULL;
		void* address_to_copy;
		size_t section_size = 0;

		scn = extract_elf_section(elfP, secHdr, desc);

		if (scn) {
			if ((data = elf_getdata(scn, NULL)) == NULL) {
				return STATUS_UNKNOWN;
			}
			section_size = data->d_size;
			if (section_size > 0) {
				address_to_copy = malloc(section_size);
				memcpy(address_to_copy, data->d_buf, section_size);
			}
		}

		if ((!scn || section_size == 0)) {
			return STATUS_UNKNOWN;
		}

		/* Create a section header */
		brig_module->section[section_id] =
				(hsa_ext_brig_section_header_t*) address_to_copy;

		return STATUS_SUCCESS;
	}

	static Elf_Scn* extract_elf_section(Elf *elfP, Elf_Data *secHdr,
			const section_desc* desc) {
		int cnt = 0;
		Elf_Scn* scn = NULL;
		Elf32_Shdr* shdr = NULL;
		char* sectionName = NULL;

		/* Iterate thru the elf sections */
		for (cnt = 1, scn = NULL; (scn = elf_nextscn(elfP, scn)); cnt++) {
			if (((shdr = elf32_getshdr(scn)) == NULL)) {
				return NULL;
			}
			sectionName = (char *) secHdr->d_buf + shdr->sh_name;
			if (sectionName
					&& ((strcmp(sectionName, desc->brig_name) == 0)
							|| (strcmp(sectionName, desc->bif_name) == 0))) {
				return scn;
			}
		}

		return NULL;
	}

	const section_desc* get_section_desc(int section_id) {
		static section_desc section_descs[] = { { SECTION_HSA_DATA, "hsa_data",
				".brig_hsa_data" }, { SECTION_HSA_CODE, "hsa_code",
				".brig_hsa_code" }, { SECTION_HSA_OPERAND, "hsa_operand",
				".brig_hsa_operand" }, };

		static const int NUM_PREDEFINED_SECTIONS = sizeof(section_descs)
				/ sizeof(section_descs[0]);
		for (int i = 0; i < NUM_PREDEFINED_SECTIONS; ++i) {
			if (section_descs[i].section_id == section_id) {
				return &section_descs[i];
			}
		}
		return NULL;
	}

};

}
}

#endif /* VIENNACL_HSA_BRIG_HELPER_HPP_ */
