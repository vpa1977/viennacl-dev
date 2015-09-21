# Targets using CPU-based execution
foreach(bench dense_blas scheduler)
   add_executable(${bench}-bench-cpu ${bench}.cpp)
endforeach()

if (ENABLE_UBLAS)
    include_directories(${Boost_INCLUDE_DIRS})
    foreach(bench sparse solver direct_solve)
      add_executable(${bench}-bench-cpu ${bench}.cpp)
      target_link_libraries(${bench}-bench-cpu ${Boost_LIBRARIES})
    endforeach()
endif (ENABLE_UBLAS)


if (ENABLE_HSA) 
 foreach(bench hsa)
    add_executable(${bench}-bench-hsa ${bench}.cpp)
    target_link_libraries(${bench}-bench-hsa ${OPENCL_LIBRARIES})
    set_target_properties(${bench}-bench-hsa PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
     target_link_libraries(${bench}-bench-hsa ${HSA_LIBRARY})
     target_link_libraries(${bench}-bench-hsa ${HSA_LIBRARY_EXT})
     target_link_libraries(${bench}-bench-hsa ${HSA_KMT})
     set_target_properties(${bench}-bench-hsa PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL -DVIENNACL_WITH_HSA")
  endforeach()
endif(ENABLE_HSA)
# Targets using OpenCL
if (ENABLE_OPENCL)

  foreach(bench dense_blas opencl)
    add_executable(${bench}-bench-opencl ${bench}.cpp)
    target_link_libraries(${bench}-bench-opencl ${OPENCL_LIBRARIES})
    set_target_properties(${bench}-bench-opencl PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
    if (ENABLE_HSA)
      target_link_libraries(${bench}-bench-opencl ${HSA_LIBRARY})
      target_link_libraries(${bench}-bench-opencl ${HSA_LIBRARY_EXT})
      target_link_libraries(${bench}-bench-opencl ${HSA_KMT})
      set_target_properties(${bench}-bench-opencl PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL -DVIENNACL_WITH_HSA")
    endif(ENABLE_HSA)

  endforeach()

  if (ENABLE_UBLAS)
     include_directories(${Boost_INCLUDE_DIRS})
     foreach(bench sparse solver direct_solve)
       add_executable(${bench}-bench-opencl ${bench}.cpp)
       target_link_libraries(${bench}-bench-opencl ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})
       set_target_properties(${bench}-bench-opencl PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
     endforeach()
  endif (ENABLE_UBLAS)

endif (ENABLE_OPENCL)

# Targets using CUDA
if (ENABLE_CUDA)

  foreach(bench dense_blas direct_solve)
     cuda_add_executable(${bench}-bench-cuda ${bench}.cu)
  endforeach()

  if (ENABLE_UBLAS)
     include_directories(${Boost_INCLUDE_DIRS})
     foreach(bench sparse solver)
       cuda_add_executable(${bench}-bench-cuda ${bench}.cu)
      target_link_libraries(${bench}-bench-cuda ${Boost_LIBRARIES})
     endforeach()
  endif (ENABLE_UBLAS)

endif (ENABLE_CUDA)


# IF(CMAKE_COMPILER_IS_GNUCXX)
   #ADD_DEFINITIONS(-Wall -pedantic -O0 -g)
#   ADD_DEFINITIONS(-Wall -pedantic -O3)
# ENDIF(CMAKE_COMPILER_IS_GNUCXX)
