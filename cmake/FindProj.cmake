# - Find proj
# Find the native proj headers and libraries.
#
# PROJ_INCLUDE_DIR	- where to find proj_api.h, etc.
# PROJ_LIBRARIES	- List of libraries when using proj.
# PROJ_FOUND	- True if proj found.

# Look for the header file.
find_path(PROJ_INCLUDE_DIR NAMES proj_api.h)

# Look for the library.
find_library(PROJ_LIBRARIES NAMES proj)

# Handle the QUIETLY and REQUIRED arguments and set PROJ_FOUND to TRUE if all listed variables are TRUE.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PROJ DEFAULT_MSG PROJ_LIBRARIES PROJ_INCLUDE_DIR)

mark_as_advanced(PROJ_INCLUDE_DIR PROJ_LIBRARIES)
