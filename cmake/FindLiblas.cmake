# - Find liblas
# Find the native liblas headers and libraries.
#
# LIBLAS_INCLUDE_DIR	- where to find liblas.h, etc.
# LIBLAS_LIBRARIES	- List of libraries when using liblas.
# LIBLAS_VERSION	- liblas version if found
# LIBLAS_FOUND	- True if liblas found.

# Look for the header file.
find_path(LIBLAS_INCLUDE_DIR NAMES liblas/capi/liblas.h liblas/capi/las_version.h liblas/capi/las_config.h)

# Look for the library.
find_program(LIBLAS_CONFIG "liblas-config")
if(LIBLAS_CONFIG)
	exec_program("${LIBLAS_CONFIG}" ARGS "--libs" OUTPUT_VARIABLE LIBLAS_LIBRARIES RETURN_VALUE LIBLAS_LIBDIR_CODE)
	if(NOT LIBLAS_LIBDIR_CODE EQUAL 0)
		unset(LIBLAS_LIBRARIES)
	endif()
endif()

# Handle the QUIETLY and REQUIRED arguments and set LIBLAS_FOUND to TRUE if all listed variables are TRUE.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LIBLAS DEFAULT_MSG LIBLAS_LIBRARIES LIBLAS_INCLUDE_DIR)

if(LIBLAS_FOUND)
	file(STRINGS "${LIBLAS_INCLUDE_DIR}/liblas/capi/las_version.h" LIBLAS_VERSION_LINES REGEX "#define[ \t]+LIBLAS_VERSION_(MAJOR|MINOR|REV)")
	string(REGEX REPLACE ".*LIBLAS_VERSION_MAJOR *\([0-9]*\).*" "\\1" LIBLAS_VERSION_MAJOR "${LIBLAS_VERSION_LINES}")
	string(REGEX REPLACE ".*LIBLAS_VERSION_MINOR *\([0-9]*\).*" "\\1" LIBLAS_VERSION_MINOR "${LIBLAS_VERSION_LINES}")
	string(REGEX REPLACE ".*LIBLAS_VERSION_REV *\([0-9]*\).*" "\\1" LIBLAS_VERSION_REV "${LIBLAS_VERSION_LINES}")
	set(LIBLAS_VERSION "${LIBLAS_VERSION_MAJOR}.${LIBLAS_VERSION_MINOR}.${LIBLAS_VERSION_REV}")
endif()

mark_as_advanced(LIBLAS_INCLUDE_DIR LIBLAS_LIBRARIES LIBLAS_VERSION)
