# - Find uuid
# Find the native uuid headers and libraries.
#
# UUID_INCLUDE_DIR	- where to find uuid.h, etc.
# UUID_LIBRARIES	- List of libraries when using uuid.
# UUID_FOUND	- True if uuid found.

# Look for the header file.
find_path(UUID_INCLUDE_DIR NAMES uuid/uuid.h)

# Look for the library.
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux") # Linux requires a separate library for UUID
	find_library(UUID_LIBRARIES NAMES uuid)
else()
	set(UUID_LIBRARIES "")
endif()

cmake_push_check_state()
set(CMAKE_REQUIRED_INCLUDES "${CMAKE_REQUIRED_INCLUDES};${UUID_INCLUDE_DIR}")
set(CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES};${UUID_LIBRARIES}")
check_symbol_exists("uuid_generate" "uuid/uuid.h" HAVE_UUID_GENERATE)
cmake_pop_check_state()
if(NOT HAVE_UUID_GENERATE)
	set(UUID_FOUND OFF)
endif()

# Handle the QUIETLY and REQUIRED arguments and set UUID_FOUND to TRUE if all listed variables are TRUE.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(UUID DEFAULT_MSG UUID_LIBRARIES UUID_INCLUDE_DIR)

mark_as_advanced(UUID_INCLUDE_DIR UUID_LIBRARIES)
