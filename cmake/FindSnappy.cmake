# - Find snappy
# Find the native snappy headers and libraries.
#
# SNAPPY_INCLUDE_DIR	- where to find snappy.h, etc.
# SNAPPY_LIBRARIES	- List of libraries when using snappy.
# SNAPPY_FOUND	- True if snappy found.

# Look for the header file.
find_path(SNAPPY_INCLUDE_DIR NAMES snappy.h)

# Look for the library.
find_library(SNAPPY_LIBRARIES NAMES snappy)

# Handle the QUIETLY and REQUIRED arguments and set SNAPPY_FOUND to TRUE if all listed variables are TRUE.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SNAPPY DEFAULT_MSG SNAPPY_LIBRARIES SNAPPY_INCLUDE_DIR)

mark_as_advanced(SNAPPY_INCLUDE_DIR SNAPPY_LIBRARIES)
