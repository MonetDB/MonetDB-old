# - Find iconv
# Test if iconv is natively supported or not
# WARNING This file name is FindIconvMonetDB to not confuse with FindIconv from Cmake. We don't use that version because
# we have to check for existence of libiconv_open symbol in FreeBSD.
#
# ICONV_INCLUDE_DIR	- where to find iconv.h, etc.
# ICONV_LIBRARIES	- List of libraries when using iconv.
# ICONV_FOUND	- True if iconv found.
# ICONV_IS_BUILT_IN - If iconv is built in

if(${CMAKE_SYSTEM_NAME} STREQUAL "FreeBSD") # On FreeBSD, libiconv_open symbol is required
	check_symbol_exists("libiconv_open" "iconv.h" ICONV_IS_BUILT_IN)
else()
	check_symbol_exists("iconv_open" "iconv.h" ICONV_IS_BUILT_IN)
endif()

if(ICONV_IS_BUILT_IN)
	set(ICONV_INCLUDE_DIR "")
	set(ICONV_LIBRARY_NAME "c")
else()
	# Look for the header file
	find_path(ICONV_INCLUDE_DIR NAMES "iconv.h" DOC "iconv include directory")
	# Search if the library name is iconv or libiconv
	find_path(ICONV_LIBRARY_NAME NAMES "iconv" "libiconv" DOC "iconv library")
endif()

# Look for the library
find_library(ICONV_LIBRARIES NAMES "${ICONV_LIBRARY_NAME}" DOC "iconv library (potentially the C library)")

include(FindPackageHandleStandardArgs)
if(ICONV_IS_BUILT_IN) # If the library is built in, we don't have to check if the include directory is ok, because it will be empty
	find_package_handle_standard_args(Iconv REQUIRED_VARS ICONV_LIBRARIES)
else()
	find_package_handle_standard_args(Iconv REQUIRED_VARS ICONV_LIBRARIES ICONV_INCLUDE_DIR)
endif()

mark_as_advanced(ICONV_INCLUDE_DIR ICONV_LIBRARIES)
