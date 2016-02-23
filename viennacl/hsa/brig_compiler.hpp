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

    /** @brief
     Invoke CLOC https://github.com/HSAFoundation/CLOC offline compiler
     */
    struct compiler_helper
    {
      const char* CLOC_COMPILER;
      
      compiler_helper() : CLOC_COMPILER(0)
      {
        setenv("CLOC_COMPILER", "/home/bsp/CLOC/bin/cloc.sh", 0);
        CLOC_COMPILER = getenv("CLOC_COMPILER");
      }

      std::vector<char> compile_brig(const std::string& content)
      {
        char buffer[L_tmpnam];
        strcpy(buffer, "baseXXXXXX");
        mktemp(buffer);

        std::string name(buffer);
        name += ".cl";
        FILE* tmp = fopen(name.c_str(), "wb+");
        fwrite(content.c_str(), content.size(), 1, tmp);
        fclose(tmp);
        std::string command(CLOC_COMPILER);
        command += " ";
        command += name;
        int err = system(command.c_str());
        remove(name.c_str());
        name = buffer;
        name += ".brig";
        if (err)
          throw "CLOC compilation failed";

        tmp = fopen(name.c_str(), "rb");
        fseek(tmp, 0, SEEK_END);
        std::vector<char> binary;
        binary.resize(ftell(tmp));
        rewind(tmp);
        err = fread(&binary[0], binary.size(), 1, tmp);
        fclose(tmp);
        remove(name.c_str());
        return binary;


      }

    };
  }
}

#endif
