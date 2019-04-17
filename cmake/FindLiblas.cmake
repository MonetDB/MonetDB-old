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
find_library(LIBLAS_LIBRARY las)
find_library(LIBLAS_C_LIBRARY las_c)
find_library(BOOST_PROGRAM_LIBRARY boost_program_options)
find_library(BOOST_THREAD_LIBRARY boost_thread)
find_library(GDL_LIBRARY gdal)
find_library(GEOTIFF_LIBRARY geotiff)
find_library(TIFF_LIBRARY tiff)
find_library(LASZIP_LIBRARY laszip)
if(LIBLAS_LIBRARY AND LIBLAS_C_LIBRARY AND BOOST_PROGRAM_LIBRARY AND BOOST_THREAD_LIBRARY AND GDL_LIBRARY AND GEOTIFF_LIBRARY AND TIFF_LIBRARY AND LASZIP_LIBRARY)
	set(LIBLAS_LIBRARIES "${LIBLAS_LIBRARY};${LIBLAS_C_LIBRARY};${BOOST_PROGRAM_LIBRARY};${BOOST_THREAD_LIBRARY};${GDL_LIBRARY};${GEOTIFF_LIBRARY};${TIFF_LIBRARY};${LASZIP_LIBRARY}")
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
