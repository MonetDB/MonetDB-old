# Adapted from https://github.com/Framstag/libosmscout/pull/251/files#diff-40f8e3cdfc36152528d5235258332447
# Define macro to check large file support
#
# This cmake macro defines the C macro _FILE_OFFSET_BITS to 64, if it is the case.
#
# Adapted from Gromacs project (http://www.gromacs.org/) by Julien Malik and Pedro Ferreira

macro(OPJ_TEST_LARGE_FILES)
	if(NOT FILE64_OK)
		message(STATUS "Checking for 64-bit off_t")
		# First check without any special flags
		try_compile(FILE64_OK "${PROJECT_BINARY_DIR}" "${CMAKE_MODULE_PATH}/TestFileOffsetBits.c")
		if(FILE64_OK)
			message(STATUS "Checking for 64-bit off_t - present")
		else()
			# Test with _FILE_OFFSET_BITS=64. In the future we might have 128-bit filesystems (ZFS), so it might be
			# dangerous to indiscriminately set _FILE_OFFSET_BITS=64.
			try_compile(FILE64_OK "${PROJECT_BINARY_DIR}" "${CMAKE_MODULE_PATH}/TestFileOffsetBits.c"
						COMPILE_DEFINITIONS "-D_FILE_OFFSET_BITS=64")
			if(FILE64_OK)
				set(_FILE_OFFSET_BITS 64 CACHE INTERNAL "Result of test for needed _FILE_OFFSET_BITS=64")
				message(STATUS "Checking for 64-bit off_t - present with _FILE_OFFSET_BITS=64")
			else()
				message(STATUS "Checking for 64-bit off_t - not present")
			endif()
		endif()
	endif()
endmacro()
