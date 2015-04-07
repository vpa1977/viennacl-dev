#ifndef VIENNACL_HSA_COMPILER_HPP_
#define VIENNACL_HSA_COMPILER_HPP_

#include <stdio.h>
#include <string.h>
#include <string>
#include <sstream>
#include <vector>
#include <stdlib.h>


namespace viennacl
{
	namespace hsa
	{
		struct compiler_helper
		{
			compiler_helper() : CLOC_COMPILER(0), CLOC_HLC(0), HSAIL_ASM(0)
			{
				setenv("CLOC_COMPILER","/home/bsp/CLOC/bin/cloc.sh", 0);
				setenv("HSAIL_ASM","/home/bsp/CLOC/bin/hsailasm", 0);
				setenv("CLOC_HLC","-p /home/bsp/HSAIL-HLC-Stable/bin", 0);

				CLOC_COMPILER = getenv("CLOC_COMPILER");
				CLOC_HLC = getenv("CLOC_HLC");
				HSAIL_ASM = getenv("HSAIL_ASM");
			}

			std::string replace(std::string subject, const std::string& search,
			                          const std::string& replace) {
			    size_t pos = 0;
			    while((pos = subject.find(search, pos)) != std::string::npos) {
			         subject.replace(pos, search.length(), replace);
			         pos += replace.length();
			    }
			    return subject;
			}

			std::string fixup(const char* line)
			{
				std::string result(line);
				result = replace( result, "activelaneshuffle", "activelanepermute");
				if (result.find("memfence_", 0) != std::string::npos)
					result = replace(result, "_group(wg)", "_wg");
				return result;
			}

			std::vector<char> compile_brig(const std::string& content, const char* program_name = NULL)
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
				command += "-hsail";
				command += " ";
				command += name;
				system(command.c_str());
				remove(name.c_str());

				name = buffer;
				name += ".hsail";
				std::stringstream hsail;


				// append module header and remove version directive - 1.0p to 1.0f fixup
				std::vector<char> line;
				line.resize(1024);
				tmp = fopen(name.c_str(), "r");

				fgets(&line[0], line.size(), tmp);

				hsail << "module &";
				hsail << (program_name == NULL ? buffer : program_name);
				hsail << ":1:0:$full:$large:$default;\n" << std::endl;

				while (fgets(&line[0], line.size(), tmp))
				{
					hsail << fixup(&line[0]);
				}

				fclose(tmp);
				std::string result = hsail.str();

				tmp = fopen(name.c_str(), "wb");
				fwrite(result.c_str(), result.size(),1, tmp);
				fclose(tmp);

				command = HSAIL_ASM;
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
			char* HSAIL_ASM;
		};
	}
};

#endif
