#ifndef VIENNACL_HSA_COMPILER_HPP_
#define VIENNACL_HSA_COMPILER_HPP_

#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include <stdlib.h>

namespace viennacl
{
	namespace hsa
	{
		struct compiler_helper
		{
			compiler_helper() : CLOC_COMPILER(0), CLOC_HLC(0)
			{
				setenv("CLOC_COMPILER","/home/bsp/CLOC/bin/cloc", 0);
				setenv("CLOC_HLC","-p1 /home/bsp/HSAIL-HLC-Stable/bin", 0);

				CLOC_COMPILER = getenv("CLOC_COMPILER");
				CLOC_HLC = getenv("CLOC_HLC");
			}

			std::vector<char> compile_brig(const std::string& content)
			{
				char buffer[L_tmpnam];
				strcpy(buffer,"baseXXXXXX");
				mktemp(buffer);

				std::string name(buffer);
				name+= ".cl";
				FILE* tmp = fopen(name.c_str(), "wb+");
				fwrite(content.c_str(), content.size(), 1, tmp);
				fclose(tmp);
				std::string command(CLOC_COMPILER);
				command += " ";
				command += CLOC_HLC;
				command += " ";
				command += name;
				system(command.c_str());
				remove(name.c_str());

				name = buffer;
				name += ".brig";
				tmp = fopen(name.c_str(), "rb");
				fseek(tmp,0,SEEK_END);
				std::vector<char> binary;
				binary.resize(ftell(tmp));
				rewind(tmp);
				fread(&binary[0], binary.size(), 1, tmp);
				fclose(tmp);
				remove(name.c_str());
				return binary;
			}

			char* CLOC_COMPILER;
			char* CLOC_HLC;
		};
	}
};

#endif
