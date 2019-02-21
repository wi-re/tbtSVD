cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
# CMake policy settings
cmake_policy(VERSION 3.9)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
SET(Boost_USE_STATIC_LIBS        OFF)
SET(Boost_USE_MULTITHREADED      ON)
SET(Boost_USE_STATIC_RUNTIME     OFF)
SET(BOOST_ALL_DYN_LINK           ON)   
SET(CUDA_ATTACH_VS_BUILD_RULE_TO_FILE ON)
SET(CMAKE_EXPORT_COMPILE_COMMANDS ON) 
SET(CUDA_SEPARABLE_COMPILATION ON)								 		
SET(CUDA_HOST_COMPILATION_CPP ON)								
SET(CMAKE_INCLUDE_CURRENT_DIR ON)	# needed for QT 
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) 
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  
# Project settings
SET(NAME openGPUSPH)													# name of the Project, variable for convenience
project(${NAME} LANGUAGES CXX CUDA)														# set the project name using the variable

# steps to determine the GPU architecture of the system to only build one architecture
SET(OUTPUTFILE ${CMAKE_CURRENT_BINARY_DIR}/cuda_script) # No suffix required
SET(CUDAFILE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_cuda.cu)
execute_process(COMMAND nvcc -lcuda ${CUDAFILE} -o ${OUTPUTFILE})
execute_process(COMMAND ${OUTPUTFILE} RESULT_VARIABLE CUDA_RETURN_CODE OUTPUT_VARIABLE ARCH)
if(${CUDA_RETURN_CODE} EQUAL 0)
	SET(CUDA_SUCCESS "TRUE")
else()
	SET(CUDA_SUCCESS "FALSE")
endif() 
if (${CUDA_SUCCESS})
	message(STATUS "CUDA Architecture: ${ARCH}")
else()
    message(WARNING ${ARCH})
endif()
# Set Include paths, due to everything needing almost everything this is done here
SET(INCLUDE_DIRS
	${CMAKE_SOURCE_DIR} 		
	${CUDA_TOOLKIT_INCLUDE}	
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	${CMAKE_SOURCE_DIR} 		
	${CMAKE_BINARY_DIR}  
)
INCLUDE_DIRECTORIES(${INCLUDE_DIRS})   
# Link directories required for cmake with cuda
LINK_DIRECTORIES(${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
LINK_DIRECTORIES(${CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES})
 
 
# OS specific defines, no support for Mac OSX
IF(WIN32)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17 /EHsc /bigobj ")
  SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -gencode=arch=compute_${ARCH},code=sm_${ARCH} -gencode=arch=compute_${ARCH},code=compute_${ARCH} --expt-extended-lambda --expt-relaxed-constexpr -ftz=true -keep -use_fast_math -src-in-ptx -lineinfo -Xptxas \"-v -warn-double-usage\"")
ELSE(WIN32) 
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")  
	SET(CMAKE_CXX_STANDARD 17)
	SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
	SET(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "")
	SET(CMAKE_CUDA_FLAGS "")
	SET(CMAKE_CUDA_FLAGS_RELEASE "")
	SET(cuda_flags "-O3 -gencode=arch=compute_${ARCH},code=sm_${ARCH} -gencode=arch=compute_${ARCH},code=compute_${ARCH} --expt-extended-lambda --expt-relaxed-constexpr -keep -src-in-ptx -use_fast_math -lineinfo -Xptxas -warn-double-usage -g -DNDEBUG -std=c++14 -Xcompiler -std=c++14 --compiler-options \"-Wall -Wextra\"")
	SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${cuda_flags}")
ENDIF(WIN32)   

# Macro useful to gather all sources for cuda, c++ and qt code and sort them in the correct source group
file(GLOB_RECURSE CUDA_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} source/*.cu)
file(GLOB_RECURSE CUDA_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} source/*.cuh)
file(GLOB_RECURSE CPP_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} source/*.cpp)
file(GLOB_RECURSE CPP_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} source/*.h source/*.hpp)	  
SET(source_list ${CUDA_SOURCES} ${CUDA_HEADERS} ${CPP_SOURCES} ${CPP_HEADERS})
   
foreach(source IN LISTS source_list)
	get_filename_component(source_path "${source}" PATH) 
	string(REPLACE "/" "\\" source_path_msvc "${source_path}")
	source_group("${source_path_msvc}" FILES "${source}")
endforeach()  
  
IF(WIN32)
		add_custom_command(
			OUTPUT dummy.h
			COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cmake/dummy.h ${CMAKE_CURRENT_SOURCE_DIR}/source/dummy.h
			DEPENDS ${CUDA_SOURCES})
ENDIF()

add_executable(testSVD ${CPP_SOURCES} ${CPP_HEADERS} ${CUDA_SOURCES} ${CUDA_HEADERS})

target_link_libraries(testSVD
	${CUDA_CUDART_LIBRARY}
	${CUDA_LIBRARIES}
	${CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES}
	${CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES}	
) 