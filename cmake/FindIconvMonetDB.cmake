# - Find iconv
# Test if iconv is natively supported or not
# WARNING This file name is FindIconvMonetDB to not confuse with FindIconv from Cmake. We don't use that version because
# we have to check for existence of libiconv_open symbol in FreeBSD.
#
# ICONV_INCLUDE_DIR	- where to find iconv.h, etc.
# ICONV_LIBRARIES	- List of libraries when using iconv.
# ICONV_FOUND	- True if iconv found.
# ICONV_IS_BUILT_IN - If iconv is built in

# On FreeBSD, libiconv_open symbol is required and not supplied by the standard C library, so force the library search
if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "FreeBSD")
	check_symbol_exists("iconv_open" "iconv.h" ICONV_IS_BUILT_IN)
endif()

if(ICONV_IS_BUILT_IN)
	set(ICONV_INCLUDE_DIR "" CACHE INTERNAL "iconv include directories path")
	set(ICONV_LIBRARIES "" CACHE INTERNAL "iconv libraries path")
	set(ICONV_FOUND ON CACHE INTERNAL "iconv is available")
else()
	# Look for the header file and library
	find_path(ICONV_INCLUDE_DIR NAMES "iconv.h" DOC "iconv include directory")
	find_library(ICONV_LIBRARIES NAMES "iconv" DOC "iconv library")

	include(FindPackageHandleStandardArgs)
	find_package_handle_standard_args(Iconv REQUIRED_VARS ICONV_LIBRARIES ICONV_INCLUDE_DIR)
endif()

mark_as_advanced(ICONV_INCLUDE_DIR ICONV_LIBRARIES)
