# - Find lz4
# Find the native lz4 headers and libraries.
#
# LZ4_INCLUDE_DIR	- where to find lz4.h, etc.
# LZ4_LIBRARIES	- List of libraries when using lz4.
# LZ4_VERSION	- LZ4_VERSION if found
# LZ4_FOUND	- True if lz4 found.

# Look for the header file.
find_path(LZ4_INCLUDE_DIR NAMES lz4.h)

# Look for the library.
find_library(LZ4_LIBRARIES NAMES lz4)

# Handle the QUIETLY and REQUIRED arguments and set LZ4_FOUND to TRUE if all listed variables are TRUE.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LZ4 DEFAULT_MSG LZ4_LIBRARIES LZ4_INCLUDE_DIR)

if(LZ4_FOUND)
	file(STRINGS "${LZ4_INCLUDE_DIR}/lz4.h" LZ4_VERSION_LINES REGEX "#define[ \t]+LZ4_VERSION_(MAJOR|MINOR|RELEASE)")
	string(REGEX REPLACE ".*LZ4_VERSION_MAJOR *\([0-9]*\).*" "\\1" LZ4_VERSION_MAJOR "${LZ4_VERSION_LINES}")
	string(REGEX REPLACE ".*LZ4_VERSION_MINOR *\([0-9]*\).*" "\\1" LZ4_VERSION_MINOR "${LZ4_VERSION_LINES}")
	string(REGEX REPLACE ".*LZ4_VERSION_RELEASE *\([0-9]*\).*" "\\1" LZ4_VERSION_RELEASE "${LZ4_VERSION_LINES}")
	set(LZ4_VERSION "${LZ4_VERSION_MAJOR}.${LZ4_VERSION_MINOR}.${LZ4_VERSION_RELEASE}")
endif()

mark_as_advanced(LZ4_INCLUDE_DIR LZ4_LIBRARIES LZ4_VERSION)
