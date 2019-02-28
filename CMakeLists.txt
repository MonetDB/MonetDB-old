cmake_minimum_required(VERSION 3.10)

project(MonetDB)

set(C_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)

find_package(PkgConfig REQUIRED)
find_package(BISON REQUIRED)

set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(GNUInstallDirs REQUIRED)
include(CheckIncludeFiles REQUIRED)
include(CheckFunctionExists REQUIRED)
include(CheckTypeSize REQUIRED)
include(TestLargeFiles REQUIRED)

function(defineCmacro OPTION VALUE)
	set(${OPTION} "#define ${VALUE}" PARENT_SCOPE)
endfunction()

function(undefineCmacro OPTION)
	set(${OPTION} "#undef ${OPTION}" PARENT_SCOPE)
endfunction()

# set host data
string(TOLOWER "${CMAKE_SYSTEM_NAME}" CMAKE_SYSTEM_NAME_LOWER)
string(TOLOWER "${CMAKE_C_COMPILER_ID}" CMAKE_C_COMPILER_ID_LOWER)
defineCmacro("HOST" "HOST \"${CMAKE_SYSTEM_PROCESSOR}-pc-${CMAKE_SYSTEM_NAME_LOWER}-${CMAKE_C_COMPILER_ID_LOWER}\"")

# Read libversions file
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/libversions" LIB_VERSIONS)

string(REGEX MATCH "GDK_VERSION=[^\ \t\r\n]+" GDK_VERSION ${LIB_VERSIONS})
string(LENGTH ${GDK_VERSION} GDK_VERSION_LENGTH)
string(SUBSTRING ${GDK_VERSION} 12 GDK_VERSION_LENGTH GDK_VERSION)

string(REGEX MATCH "MAPI_VERSION=[^\ \t\r\n]+" MAPI_VERSION ${LIB_VERSIONS})
string(LENGTH ${MAPI_VERSION} MAPI_VERSION_LENGTH)
string(SUBSTRING ${MAPI_VERSION} 13 MAPI_VERSION_LENGTH MAPI_VERSION)

string(REGEX MATCH "MONETDB5_VERSION=[^\ \t\r\n]+" MONETDB5_VERSION ${LIB_VERSIONS})
string(LENGTH ${MONETDB5_VERSION} MONETDB5_VERSION_LENGTH)
string(SUBSTRING ${MONETDB5_VERSION} 17 MONETDB5_VERSION_LENGTH MONETDB5_VERSION)

string(REGEX MATCH "STREAM_VERSION=[^\ \t\r\n]+" STREAM_VERSION ${LIB_VERSIONS})
string(LENGTH ${STREAM_VERSION} STREAM_VERSION_LENGTH)
string(SUBSTRING ${STREAM_VERSION} 15 STREAM_VERSION_LENGTH STREAM_VERSION)

# Intel compiler hack
if("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -no-gcc")
endif()

# Set default build options
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vertoo.data")
	set(DFT_STRICT "YES")
	set(DFT_ASSERT "YES")
	set(DFT_DEBUG "YES")
	set(DFT_OPTIMIZE "NO")
	set(DFT_DEVELOPER "YES")
else()
	set(DFT_STRICT "NO")
	set(DFT_ASSERT "NO")
	set(DFT_DEBUG "NO")
	set(DFT_OPTIMIZE "NO")
	set(DFT_DEVELOPER "NO")
endif()

set(ENABLE_DEVELOPER "${DFT_DEVELOPER}" CACHE STRING "Build and install programs that are only useful for MonetDB development (default=yes for development sources)")
set(ENABLE_TESTING "AUTO" CACHE STRING "Enable support for testing (default=auto)")

# modules to use
set(ENABLE_MAPI "YES" CACHE STRING "Enable MAPI (default=yes)")
if(${ENABLE_MAPI} STREQUAL "NO")
	undefineCmacro(HAVE_MAPI)
else()
	defineCmacro(HAVE_MAPI "HAVE_MAPI 1")
endif()

set(ENABLE_ODBC "AUTO" CACHE STRING "Compile the MonetDB ODBC driver (default=auto)")

set(ENABLE_GDK "YES" CACHE STRING "Enable support for GDK (default=yes)")

set(ENABLE_MONETDB5 "YES" CACHE STRING "Enable support for MonetDB5 (default=yes)")
if(${ENABLE_GDK} STREQUAL "NO" AND (${ENABLE_MONETDB5} STREQUAL "YES" OR ${ENABLE_MONETDB5} STREQUAL "AUTO"))
	message(FATAL_ERROR "MonetDB5 requires GDK.")
endif()

set(ENABLE_SQL "YES" CACHE STRING "Enable support for MonetDB/SQL (default=yes)")
if(${ENABLE_MONETDB5} STREQUAL "NO" AND (${ENABLE_SQL} STREQUAL "YES" OR ${ENABLE_SQL} STREQUAL "AUTO"))
	message(FATAL_ERROR "MonetDB/SQL requires MonetDB5.")
endif()

set(ENABLE_GEOM "AUTO" CACHE STRING "Enable support for geom module (default=auto)")
if(${ENABLE_MONETDB5} STREQUAL "NO" AND (${ENABLE_GEOM} STREQUAL "YES" OR ${ENABLE_GEOM} STREQUAL "AUTO"))
	message(FATAL_ERROR "geom module requires MonetDB5.")
endif()

set(ENABLE_FITS "AUTO" CACHE STRING "Enable support for FITS (default=auto)")
set(ENABLE_NETCDF "AUTO" CACHE STRING "Enable support for netcdf (default=auto)")
set(ENABLE_LIDAR "AUTO" CACHE STRING "Enable support for LiDAR data (default=auto)")
set(ENABLE_SHP "AUTO" CACHE STRING "Enable support for ESRI Shapefiles (default=auto)")
if(${ENABLE_GEOM} STREQUAL "NO" AND (${ENABLE_SHP} STREQUAL "YES" OR ${ENABLE_SHP} STREQUAL "AUTO"))
	message(FATAL_ERROR "ESRI Shapefile vault requires the geom module.")
endif()

set(ENABLE_EMBEDDED "NO" CACHE STRING "Enable support for running MonetDB as a library (default=no)")
if(${ENABLE_EMBEDDED} STREQUAL "NO")
	undefineCmacro("HAVE_EMBEDDED")
else()
	defineCmacro("HAVE_EMBEDDED" "HAVE_EMBEDDED 1")
endif()

set(ENABLE_RINTEGRATION "AUTO" CACHE STRING "Enable support for R integration into MonetDB (default=auto)")
set(ENABLE_PY2INTEGRATION "AUTO" CACHE STRING "enable support for Python 2 integration into MonetDB (default=auto)")
set(ENABLE_PY3INTEGRATION "AUTO" CACHE STRING "enable support for Python 3 integration into MonetDB (default=auto)")

set(ENABLE_CONSOLE "YES" CACHE STRING "Enables direct console on the server (involves security risks) (default=yes)")
if(${ENABLE_CONSOLE} STREQUAL "NO")
	undefineCmacro("HAVE_CONSOLE")
else()
	defineCmacro("HAVE_CONSOLE" "HAVE_CONSOLE 1")
endif()

set(ENABLE_INT128 "AUTO" CACHE STRING "Enable support for 128-bit integers (default=auto)")
set(ENABLE_DEBUG "${DFT_DEBUG}" CACHE STRING "Enable full debugging (default=yes for development sources)")
set(ENABLE_ASSERT "${DFT_ASSERT}" CACHE STRING "Enable assertions in the code (default=yes for development sources)")
set(ENABLE_OPTIMIZE "${DFT_OPTIMIZE}" CACHE STRING "Enable extra optimization (default=no)")
set(ENABLE_STRICT "${DFT_STRICT}" CACHE STRING "Enable strict compiler flags (default=yes for development sources)")
set(ENABLE_SANITIZER "NO" CACHE STRING "Enable support for the GCC address sanitizer (default=no)")
set(ENABLE_STATIC_ANALYSIS "NO" CACHE STRING "Configure for static code analysis (use only if you know what you are doing)")
if(${ENABLE_STATIC_ANALYSIS} STREQUAL "NO")
	undefineCmacro("STATIC_CODE_ANALYSIS")
else()
	defineCmacro("STATIC_CODE_ANALYSIS" "STATIC_CODE_ANALYSIS 1")
endif()

# password hash algorithm
set(PASSWORD_BACKEND "SHA512" CACHE STRING "Passwor	d hash algorithm, one of MD5, SHA1, RIPEMD160, SHA224, SHA256, SHA384, SHA512, defaults to SHA512")
if(${PASSWORD_BACKEND} MATCHES "^MD5|SHA1|RIPEMD160|SHA224|SHA256|SHA384|SHA512$")
	defineCmacro("MONETDB5_PASSWDHASH" "MONETDB5_PASSWDHASH ${PASSWORD_BACKEND}")
	defineCmacro("MONETDB5_PASSWDHASH_TOKEN" "MONETDB5_PASSWDHASH_TOKEN \"${PASSWORD_BACKEND}\"")
else()
	message(FATAL_ERROR "PASSWORD_BACKEND invalid, choose one of MD5, SHA1, RIPEMD160, SHA224, SHA256, SHA384, SHA512")
endif()

# instalation directories
set(LOGDIR "${CMAKE_INSTALL_FULL_LOCALSTATEDIR}/log/monetdb" CACHE STRING "Where to put log files (LOCALSTATEDIR/log/monetdb/)")
set(RUNDIR "${CMAKE_INSTALL_FULL_LOCALSTATEDIR}/run/monetdb" CACHE STRING "Where to put pid files (LOCALSTATEDIR/run/monetdb/)")

OPJ_TEST_LARGE_FILES(_LARGE_FILES)
if(_LARGE_FILES)
	defineCmacro("_LARGE_FILES" "HAVE _LARGE_FILES 1")
else()
	undefineCmacro("_LARGE_FILES")
endif()
OPJ_TEST_LARGE_FILES(_LARGEFILE_SOURCE)
if(_LARGEFILE_SOURCE)
	defineCmacro("_LARGEFILE_SOURCE" "HAVE _LARGEFILE_SOURCE 1")
else()
	undefineCmacro("_LARGEFILE_SOURCE")
endif()
OPJ_TEST_LARGE_FILES(_FILE_OFFSET_BITS)
if(_FILE_OFFSET_BITS)
	defineCmacro("_FILE_OFFSET_BITS" "HAVE _FILE_OFFSET_BITS 64")
else()
	undefineCmacro("_FILE_OFFSET_BITS")
endif()

if(NOT "${CMAKE_C_COMPILER_ID}" MATCHES "^GNU|Intel|Clang|MSVC|AppleClang|SunPro$")
	message(WARNING "Compiler potentially not (correctly) recognized")
endif()

CHECK_TYPE_SIZE(int SIZEOF_INT BUILTIN_TYPES_ONLY LANGUAGE C)
CHECK_TYPE_SIZE(short SIZEOF_SHORT BUILTIN_TYPES_ONLY LANGUAGE C)
CHECK_TYPE_SIZE(char SIZEOF_CHAR BUILTIN_TYPES_ONLY LANGUAGE C)
CHECK_TYPE_SIZE(long SIZEOF_LONG BUILTIN_TYPES_ONLY LANGUAGE C)

configure_file(${CMAKE_SOURCE_DIR}/cmake/monetdb_config.h.in ${CMAKE_SOURCE_DIR}/debugme.h)

if(${ENABLE_STRICT} STREQUAL "YES")

endif()
