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

if(UUID_LIBRARIES)
	cmake_push_check_state()
	set(CMAKE_REQUIRED_INCLUDES "${CMAKE_REQUIRED_INCLUDES};${UUID_INCLUDE_DIR}")
	set(CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES};${UUID_LIBRARIES}")
	check_symbol_exists("uuid_generate" "uuid/uuid.h" HAVE_UUID_GENERATE) # some uuid instalations don't supply this symbol
	cmake_pop_check_state()
	if(NOT HAVE_UUID_GENERATE)
		set(UUID_FOUND OFF)
	endif()
endif()

# On Linux, both library and include directory path must be set
include(FindPackageHandleStandardArgs)
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
	find_package_handle_standard_args(UUID DEFAULT_MSG UUID_LIBRARIES UUID_INCLUDE_DIR)
else()
	find_package_handle_standard_args(UUID DEFAULT_MSG UUID_INCLUDE_DIR)
endif()

mark_as_advanced(UUID_INCLUDE_DIR UUID_LIBRARIES)
