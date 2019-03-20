# - Find valgrind
# Find the native valgrind headers and libraries.
#
# VALGRIND_INCLUDE_DIR	- where to find valgrind.h, etc.
# VALGRIND_LIBRARIES	- List of libraries when using valgrind.
# VALGRIND_FOUND	- True if valgrind found.

# Look for the header file.
find_path(VALGRIND_INCLUDE_DIR NAMES valgrind.h)

# Look for the library.
find_library(VALGRIND_LIBRARIES NAMES valgrind)

# Handle the QUIETLY and REQUIRED arguments and set VALGRIND_FOUND to TRUE if all listed variables are TRUE.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VALGRIND DEFAULT_MSG VALGRIND_LIBRARIES VALGRIND_INCLUDE_DIR)

mark_as_advanced(VALGRIND_INCLUDE_DIR VALGRIND_LIBRARIES)
