# - Find uuid
# Find the native uuid headers and libraries.
#
# UUID_INCLUDE_DIR	- where to find uuid.h, etc.
# UUID_LIBRARIES	- List of libraries when using uuid.
# UUID_FOUND	- True if uuid found.

# Look for the header file.
find_path(UUID_INCLUDE_DIR NAMES uuid/uuid.h)

# Look for the library.
find_library(UUID_LIBRARIES NAMES uuid)

# Handle the QUIETLY and REQUIRED arguments and set UUID_FOUND to TRUE if all listed variables are TRUE.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(UUID DEFAULT_MSG UUID_LIBRARIES UUID_INCLUDE_DIR)

mark_as_advanced(UUID_INCLUDE_DIR UUID_LIBRARIES)
