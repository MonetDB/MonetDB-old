#[[
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
#]]

if(WIN32)
	# The $<TARGET_PDB_FILE:tgt> generator expression appears on cmake 3.1
	# FindOpenSSL scripts works fine on Windows from cmake 3.7 up
	cmake_minimum_required(VERSION 3.7)
else()
	cmake_minimum_required(VERSION 3.0)
endif()

# Change these variables between releases
set(MONETDB_VERSION_MAJOR "11")
set(MONETDB_VERSION "${MONETDB_VERSION_MAJOR}.34.0")

set(GDK_VERSION_MAJOR "18")
set(GDK_VERSION "${GDK_VERSION_MAJOR}.1.0")

set(MAPI_VERSION_MAJOR "12")
set(MAPI_VERSION "${MAPI_VERSION_MAJOR}.0.0")

set(MONETDB5_VERSION_MAJOR "27")
set(MONETDB5_VERSION "${MONETDB5_VERSION_MAJOR}.1.0")

set(STREAM_VERSION_MAJOR "13")
set(STREAM_VERSION "${STREAM_VERSION_MAJOR}.1.0")

# HOMEPAGE_URL "https://www.monetdb.org/" in project command supported from cmake 3.12 up
# DESCRIPTION "MonetDB, SQL queries at light speed" in project command supported from cmake 3.9 up
project(MonetDB VERSION "${MONETDB_VERSION}" LANGUAGES C)
set(C_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)

set(CMAKE_FIND_FRAMEWORK LAST) # We give precedence to UNIX include directories over OS X Frameworks directories
set(CMAKE_FIND_APPBUNDLE LAST)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" CACHE INTERNAL "Location of custom CMake modules.")
include(CheckCSourceCompiles REQUIRED)
include(CheckIncludeFile REQUIRED)
include(CMakePushCheckState REQUIRED)
include(CheckStructHasMember REQUIRED)
include(CheckSymbolExists REQUIRED)
include(CheckTypeSize REQUIRED)
include(TestBigEndian REQUIRED)
include(TestLargeFiles REQUIRED)

# Set host information
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" CMAKE_SYSTEM_PROCESSOR_LOWER)
string(TOLOWER "${CMAKE_SYSTEM_NAME}" CMAKE_SYSTEM_NAME_LOWER)
string(TOLOWER "${CMAKE_C_COMPILER_ID}" CMAKE_C_COMPILER_ID_LOWER)
set("HOST" "${CMAKE_SYSTEM_PROCESSOR_LOWER}-pc-${CMAKE_SYSTEM_NAME_LOWER}-${CMAKE_C_COMPILER_ID_LOWER}")

set(DFT_STRICT "NO")
# Check current version control revision
if(EXISTS "${CMAKE_SOURCE_DIR}/.hg")
	exec_program("hg" "${CMAKE_SOURCE_DIR}" ARGS "id -i" OUTPUT_VARIABLE HG_OUPUT_RES RETURN_VALUE HG_RETURN_CODE)
	if(HG_RETURN_CODE EQUAL 0 AND HG_OUPUT_RES)
		set(MERCURIAL_ID "${HG_OUPUT_RES}")
		set(DFT_STRICT "YES")
	endif()
elseif(EXISTS "${CMAKE_SOURCE_DIR}/.git")
	exec_program("git" "${CMAKE_SOURCE_DIR}" ARGS "rev-parse HEAD" OUTPUT_VARIABLE GIT_OUPUT_RES RETURN_VALUE GIT_RETURN_CODE)
	if(GIT_RETURN_CODE EQUAL 0 AND GIT_OUPUT_RES)
		set(MERCURIAL_ID "${GIT_OUPUT_RES}+")
		set(DFT_STRICT "YES")
	endif()
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}") # Initialize variable
if(NOT MSVC AND NOT ${CMAKE_C_COMPILER_ID} MATCHES "^GNU|Intel|Clang|AppleClang$")
	message(WARNING "Compiler potentially not (correctly) recognized")
endif()
if(NOT CMAKE_BUILD_TYPE) # Set default build options
	message(STATUS "Setting build type to Debug as none was selected")
	set(CMAKE_BUILD_TYPE Debug CACHE STRING "The type of build" FORCE)
endif()

# Extensions to use
set(ENABLE_CINTEGRATION "AUTO" CACHE STRING "Enable support for C UDFs (default=AUTO)") # Check with HAVE_CUDF
set(ENABLE_CONSOLE "YES" CACHE STRING "Enables direct console on the server (involves security risks) (default=YES)") # Check with HAVE_CONSOLE
set(ENABLE_EMBEDDED "NO" CACHE STRING "Enable support for running MonetDB as a library (default=NO)") # TODO check this
set(ENABLE_FITS "AUTO" CACHE STRING "Enable support for FITS (default=AUTO)") # Check with HAVE_FITS
set(ENABLE_GDK "YES" CACHE STRING "Enable support for GDK (default=YES)") # Check with HAVE_GDK
set(ENABLE_GEOM "AUTO" CACHE STRING "Enable support for geom module (default=AUTO)") # Check with HAVE_GEOM
set(ENABLE_INT128 "AUTO" CACHE STRING "Enable support for 128-bit integers (default=AUTO)") # Check with HAVE_HGE
set(ENABLE_LIDAR "AUTO" CACHE STRING "Enable support for LiDAR data (default=AUTO)") # Check with HAVE_LIDAR
set(ENABLE_MAPI "YES" CACHE STRING "Enable MAPI (default=YES)") # Check with HAVE_MAPI
set(ENABLE_MONETDB5 "YES" CACHE STRING "Enable support for MonetDB5 (default=YES)") # Check with HAVE_MONETDB5
set(ENABLE_NETCDF "AUTO" CACHE STRING "Enable support for netcdf (default=AUTO)") # Check with HAVE_NETCDF
set(ENABLE_ODBC "AUTO" CACHE STRING "Compile the MonetDB ODBC driver (default=AUTO)") # Check with HAVE_ODBC
set(ENABLE_PY3INTEGRATION "AUTO" CACHE STRING "enable support for Python 3 integration into MonetDB (default=AUTO)") # Check with HAVE_LIBPY3
set(ENABLE_RINTEGRATION "AUTO" CACHE STRING "Enable support for R integration into MonetDB (default=AUTO)") # Check with HAVE_LIBR
set(ENABLE_SAMTOOLS "AUTO" CACHE STRING "Include Samtools support (default=AUTO)") # Check with HAVE_SAMTOOLS
set(ENABLE_SANITIZER "NO" CACHE STRING "Enable support for the GCC address sanitizer (default=NO)")
set(ENABLE_SHP "AUTO" CACHE STRING "Enable support for ESRI Shapefiles (default=AUTO)") # Check with HAVE_SHP
set(ENABLE_SQL "YES" CACHE STRING "Enable support for MonetDB/SQL (default=YES)") # Check with HAVE_SQL
set(ENABLE_STATIC_ANALYSIS "NO" CACHE STRING "Configure for static code analysis (use only if you know what you are doing)") # Check with STATIC_CODE_ANALYSIS
set(ENABLE_STRICT "${DFT_STRICT}" CACHE STRING "Enable strict compiler flags (default=YES for development sources, NO for tarball instalation)")
set(ENABLE_TESTING "AUTO" CACHE STRING "Enable support for testing (default=AUTO)") # Check with HAVE_TESTING

# Optional extra libraries
set(WITH_BZ2 "AUTO" CACHE STRING "Include bz2 support (default=AUTO)") # Check with HAVE_LIBLZMA
set(WITH_CURL "AUTO" CACHE STRING "Include cURL support (default=AUTO)") # Check with HAVE_CURL
set(WITH_LIBLZMA "AUTO" CACHE STRING "Include liblzma support (default=AUTO)") # Check with HAVE_LIBLZMA
set(WITH_LIBXML2 "AUTO" CACHE STRING "Include libxml2 support (default=AUTO)") # Check with HAVE_LIBXML
set(WITH_LZ4 "AUTO" CACHE STRING "Include lz4 library support (default=AUTO)") # Check with HAVE_LIBLZ4
set(WITH_PROJ "AUTO" CACHE STRING "Include proj support (default=AUTO)") # Check with HAVE_PROJ
set(WITH_READLINE "AUTO" CACHE STRING "Include readline support (default=AUTO)") # Check with HAVE_LIBREADLINE
set(WITH_REGEX "AUTO" CACHE STRING "Use either PCRE or POSIX regular expressions (default=PCRE if available)") # Check with HAVE_LIBPCRE and HAVE_POSIX_REGEX
set(WITH_SNAPPY "AUTO" CACHE STRING "Include snappy support (default=AUTO)") # Check with HAVE_LIBSNAPPY
set(WITH_UUID "AUTO" CACHE STRING "Include UUID support (default=AUTO)") # Check with HAVE_UUID
set(WITH_VALGRIND "NO" CACHE STRING "Include valgrind support (default=NO)") # Check with HAVE_VALGRIND
set(WITH_ZLIB "AUTO" CACHE STRING "Include zlib support (default=AUTO)") # Check with HAVE_LIBZ

# Instalation directories
set(PROGRAM_PERMISSIONS_DEFAULT OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE) # required for some instalation files
if(WIN32) # Variables with _DIR are used for monetdb_config.h file, which need proper escape on Windows
	string(REPLACE "/" "\\\\" TRIMMED_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}") # Fix cmake conversions
	set(BINDIR "${CMAKE_INSTALL_PREFIX}/bin")
	set(BIN_DIR "${TRIMMED_INSTALL_PREFIX}\\\\bin")
	set(DATADIR "${CMAKE_INSTALL_PREFIX}/share")
	set(DATA_DIR "${TRIMMED_INSTALL_PREFIX}\\\\share")
	set(DATAROOTDIR "${TRIMMED_INSTALL_PREFIX}/share")
	set(INCLUDEDIR "${CMAKE_INSTALL_PREFIX}/include")
	set(INFODIR "${CMAKE_INSTALL_PREFIX}/info")
	set(LIBDIR "${CMAKE_INSTALL_PREFIX}/lib")
	set(LIB_DIR "${TRIMMED_INSTALL_PREFIX}\\\\lib")
	set(LIBEXECDIR "${CMAKE_INSTALL_PREFIX}/libexec")
	set(LOCALSTATEDIR "${CMAKE_INSTALL_PREFIX}/var")
	set(LOCALSTATE_DIR "${TRIMMED_INSTALL_PREFIX}\\\\var")
	set(MANDIR "${CMAKE_INSTALL_PREFIX}/man")
	set(SYSCONFDIR "${CMAKE_INSTALL_PREFIX}/etc")
else()
	include(GNUInstallDirs REQUIRED)
	set(BINDIR "${CMAKE_INSTALL_FULL_BINDIR}")
	set(BIN_DIR "${CMAKE_INSTALL_FULL_BINDIR}")
	set(DATADIR "${CMAKE_INSTALL_FULL_DATADIR}")
	set(DATA_DIR "${CMAKE_INSTALL_FULL_DATADIR}")
	set(DATAROOTDIR "${CMAKE_INSTALL_FULL_DATAROOTDIR}")
	set(INCLUDEDIR "${CMAKE_INSTALL_FULL_INCLUDEDIR}")
	set(INFODIR "${CMAKE_INSTALL_FULL_INFODIR}")
	set(LIBDIR "${CMAKE_INSTALL_FULL_LIBDIR}")
	set(LIB_DIR "${CMAKE_INSTALL_FULL_LIBDIR}")
	set(LIBEXECDIR "${CMAKE_INSTALL_FULL_LIBEXECDIR}")
	set(LOCALSTATEDIR "${CMAKE_INSTALL_FULL_LOCALSTATEDIR}")
	set(LOCALSTATE_DIR "${CMAKE_INSTALL_FULL_LOCALSTATEDIR}")
	set(MANDIR "${CMAKE_INSTALL_FULL_MANDIR}")
	set(SYSCONFDIR "${CMAKE_INSTALL_FULL_SYSCONFDIR}")
	# Unix only variables
	set(LOGDIR "${CMAKE_INSTALL_FULL_LOCALSTATEDIR}/log/monetdb" CACHE PATH "Where to put log files (default LOCALSTATEDIR/log/monetdb)")
	set(PKGCONFIGDIR "${LIBDIR}/pkgconfig") # Set a default value for every installation
	set(RUNDIR "${CMAKE_INSTALL_FULL_LOCALSTATEDIR}/run/monetdb" CACHE PATH "Where to put pid files (default LOCALSTATEDIR/run/monetdb)")
	set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}")
endif()

function(MT_addCompilerFlag REGEX_EXPRESSION FLAG_TO_ADD OLD_FLAGS BUILD NEW_FLAGS)
	if(NOT "${OLD_FLAGS}" MATCHES "${REGEX_EXPRESSION}")
		set(${NEW_FLAGS} "${OLD_FLAGS} ${FLAG_TO_ADD}" PARENT_SCOPE)
		message(STATUS "Added ${FLAG_TO_ADD} flag for ${BUILD} build(s)")
	endif()
endfunction()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" AND NOT ${CMAKE_C_COMPILER_ID} STREQUAL "Intel")
	add_definitions(-D_GNU_SOURCE -D_XOPEN_SOURCE) # required for GNU extensions
	set(CMAKE_REQUIRED_DEFINITIONS "${CMAKE_REQUIRED_DEFINITIONS} -D_GNU_SOURCE -D_XOPEN_SOURCE") # required for tests
endif()
if(${CMAKE_C_COMPILER_ID} STREQUAL "Intel")
	if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
		MT_addCompilerFlag("-no-gcc-include-dir" "-no-gcc-include-dir" "${CMAKE_C_FLAGS}" "all" CMAKE_C_FLAGS)
	endif()
	if(EXISTS "${ROOT}/usr/include/math.h") # see https://software.intel.com/en-us/forums/intel-c-compiler/topic/760979
		file(READ "${ROOT}/usr/include/math.h" TMPTXT)
		string(FIND "${TMPTXT}" "_LIB_VERSION_TYPE" FOUND_VERSION)
		if(NOT FOUND_VERSION)
			set(INTEL_MATH_H_HACK ON CACHE INTERNAL "must use intel math.h hack")
		endif()
	endif()
	set(NAN_CANNOT_BE_USED_AS_INITIALIZER ON CACHE INTERNAL "NaN cannot be used as an initializer") # This hack is only required by the Intel compiler
elseif(MSVC)
	set(restrict "__restrict") # C99 feature not present in MSVC
	set(inline "__inline") # C99 feature only available on C++ compiler in MSVC https://docs.microsoft.com/en-us/cpp/cpp/inline-functions-cpp?view=vs-2015
	add_definitions(/D_CRT_SECURE_NO_WARNINGS) # TODO it would be nice to remove this, but it requires lots of rewriting
endif()

# Tests for large files support
opj_test_large_files(_LARGE_FILES)
opj_test_large_files(_LARGEFILE_SOURCE)
opj_test_large_files(_FILE_OFFSET_BITS)
opj_test_large_files(_DARWIN_USE_64_BIT_INODE)
#Test if struct tm has tm_zone field
check_struct_has_member("struct tm" tm_zone "time.h" HAVE_STRUCT_TM_TM_ZONE LANGUAGE C)
#Test CPU endianness
test_big_endian(WORDS_BIGENDIAN)

#Test for __builtin_add_overflow
check_c_source_compiles("
	#include <stdio.h>
	#include <stdlib.h>

	int main(int argc, char **argv) {
		__int128 a=(__int128)atoll(getenv(\"a\"))<<64, b=(__int128)atoll(getenv(\"b\"))<<64, c;
		(void) argc;(void) argv;
		if (!__builtin_mul_overflow(a, b, &c))
			printf(\"%lld %lld\",(long long)(c>>64), (long long)c);
		return 0;
	}" HAVE___BUILTIN_ADD_OVERFLOW)

#Test if _Noreturn keyword is supported
if(MSVC)
	set(_Noreturn "__declspec(noreturn)")
else()
	check_c_source_compiles("
		#include <stdlib.h>

		_Noreturn void foo(const char *a, ...);
		void foo(const char *a, ...) { (void)a; exit(0); }

		int main(int argc, char **argv) {
			(void) argc; (void) argv;
			foo (\"%s\", \"\");
			return 0;
		}" HAVE_NORETURN_KEYWORD)
	if(NOT HAVE_NORETURN_KEYWORD)
		check_symbol_exists("__has_attribute(__noreturn__)" "" HAVE_NORETURN_ATTRIBUTE)
		if(HAVE_NORETURN_ATTRIBUTE) # If the compiler supports the attribute
			set(_Noreturn "__attribute__((__noreturn__))")
		else()
			set(_Noreturn " ") # remove keyword
		endif()
	endif()
endif()

#Test if compiler has usable <stdatomic.h>
check_c_source_compiles("
	#include <stdatomic.h>

	#if ATOMIC_LLONG_LOCK_FREE == 2
		atomic_ullong i;
	#else
		atomic_uint i;
	#endif
	typedef void *_Atomic volatile aptr;

	int main(int argc, char **argv) {
		(void) argc; (void) argv;
		atomic_store(&i, 0);
		return 0;
	}" HAVE_STDATOMIC_H)

if(WIN32)
	set(OS_DIRSEP  "\\\\")
	set(OS_PATHSEP ";")
	set(SO_PREFIX  "lib")
	# We don't use the library naming convention on Windows :(
	set(CMAKE_IMPORT_LIBRARY_PREFIX "${SO_PREFIX}" CACHE STRING "Import library name prefix" FORCE)
	set(CMAKE_SHARED_LIBRARY_PREFIX "${SO_PREFIX}" CACHE STRING "Shared library name prefix" FORCE)
	set(CMAKE_SHARED_MODULE_PREFIX "${SO_PREFIX}" CACHE STRING "Library module name prefix" FORCE)
	set(CMAKE_STATIC_LIBRARY_PREFIX "${SO_PREFIX}" CACHE STRING "Static Library name prefix" FORCE)
	set(COMPILER_OPTION "/") # Used for Intel compiler available on Windows and Linux
	set(INTEL_OPTION_EXTRA "Q") # Intel compiler extra letter on some Windows parameters
elseif(UNIX)
	set(OS_DIRSEP  "/")
	set(OS_PATHSEP ":")
	set(SO_PREFIX  "${CMAKE_SHARED_LIBRARY_PREFIX}")
	set(COMPILER_OPTION "-") # Used for Intel compiler available on Windows and Linux
else()
	message(FATAL_ERROR "Unknown target operating system")
endif()
set(DIR_SEP      "'${OS_DIRSEP}'")
set(DIR_SEP_STR  ${OS_DIRSEP})
set(PATH_SEP     "'${OS_PATHSEP}'")
set(SO_EXT       "${CMAKE_SHARED_LIBRARY_SUFFIX}")

# We need python3 for the testweb, python udfs and c udfs compilation
if(CMAKE_VERSION VERSION_LESS 3.12)
	set(Python_ADDITIONAL_VERSIONS 3.7 3.6 3.5)
	find_package(PythonInterp 3.4)
	find_package(PythonLibs 3.4)
	if(PYTHONINTERP_FOUND)
		set(Python3_Interpreter_FOUND ON)
		set(Python3_VERSION_MAJOR "${PYTHON_VERSION_MAJOR}")
		set(Python3_VERSION_MINOR "${PYTHON_VERSION_MINOR}")
		set(Python3_VERSION_PATCH "${PYTHON_VERSION_PATCH}")
		set(Python3_EXECUTABLE "${PYTHON_EXECUTABLE}")
	endif()
	if(PYTHONLIBS_FOUND)
		set(Python3_Development_FOUND ON)
		set(Python3_INCLUDE_DIRS "${PYTHON_INCLUDE_DIRS}")
		set(Python3_LIBRARIES "${PYTHON_LIBRARIES}")
	endif()
else()
	find_package(Python3 COMPONENTS Interpreter Development)
endif()
if(Python3_Interpreter_FOUND)
	set(PYTHON3_REQUIRED_VERSION "3.4")
	if("${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}" VERSION_LESS ${PYTHON3_REQUIRED_VERSION})
		message(WARNING "The Python 3 executable is too old ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR} < ${PYTHON3_REQUIRED_VERSION}")
	else()
		exec_program("${Python3_EXECUTABLE}" ARGS "-c \"import distutils.sysconfig; print(distutils.sysconfig.get_python_lib(1,0,'${LIBDIR}'))\"" OUTPUT_VARIABLE PYTHON3_LIBDIR RETURN_VALUE PYTHON3_LIBDIR_CODE)
		if(PYTHON3_LIBDIR AND PYTHON3_LIBDIR_CODE EQUAL 0)
			string(LENGTH "${LIBDIR}" LIBDIR_LENGTH)
			string(LENGTH "${PYTHON3_LIBDIR}" PYTHON3_LIBDIR_PREFIX_LENGTH)
			math(EXPR LEN1 "${LIBDIR_LENGTH}+1") # add the / separator
			string(SUBSTRING "${PYTHON3_LIBDIR}" ${LEN1} -1 PYTHON_LIBDIR) # remove the prefix
		else()
			message(WARNING "Could not determine MonetDB Python testing files instalation directory")
		endif()
		set(HAVE_PYTHON3 ON CACHE INTERNAL "python3 is available")
	endif()
endif()

# Bison
find_package(BISON)
if(NOT BISON_FOUND)
	if(${ENABLE_SQL} STREQUAL "YES")
		message(FATAL_ERROR "MonetDB/SQL requires bison")
	endif()
	set(ENABLE_SQL "NO" CACHE STRING "Bison is required for MonetDB/SQL" FORCE)
endif()

# Iconv
include(FindIconvMonetDB)
if(ICONV_FOUND)
	set(HAVE_ICONV ON CACHE INTERNAL "iconv is available")
else()
	set(ICONV_INCLUDE_DIR "" CACHE INTERNAL "iconv include directory")
	set(ICONV_LIBRARIES "" CACHE INTERNAL "iconv libraries to link")
endif()

# Thread libraries
set(THREADS_PREFER_PTHREAD_FLAG ON) # We do prefer pthreads
find_package(Threads)
set(THREAD_LIBRARIES "${CMAKE_THREAD_LIBS_INIT}")

# Password hash algorithm
set(PASSWORD_BACKEND "SHA512" CACHE STRING "Password hash algorithm, one of MD5, SHA1, RIPEMD160, SHA224, SHA256, SHA384, SHA512, defaults to SHA512")
if(${PASSWORD_BACKEND} MATCHES "^MD5|SHA1|RIPEMD160|SHA224|SHA256|SHA384|SHA512$")
	set(MONETDB5_PASSWDHASH ${PASSWORD_BACKEND})
	set(MONETDB5_PASSWDHASH_TOKEN ${PASSWORD_BACKEND})
else()
	message(FATAL_ERROR "PASSWORD_BACKEND invalid, choose one of MD5, SHA1, RIPEMD160, SHA224, SHA256, SHA384, SHA512")
endif()

# OpenSSL or CommonCrypto library
cmake_push_check_state()
if(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
	include(FindCommonCrypto)

	if(COMMONCRYPTO_FOUND)
		set(HAVE_COMMONCRYPTO ON CACHE INTERNAL "Commoncryto is available")
		set(CRYPTO_INCLUDE_DIR "${COMMONCRYPTO_INCLUDE_DIR}" CACHE INTERNAL "crypto include directory")
		set(CRYPTO_LIBRARIES "${COMMONCRYPTO_LIBRARIES}" CACHE INTERNAL "crypto libraries to link")
		set(CMAKE_REQUIRED_INCLUDES "${CMAKE_REQUIRED_INCLUDES};${CRYPTO_INCLUDE_DIR}")
		set(CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES};${CRYPTO_LIBRARIES}")
		check_symbol_exists("CC_MD5_Update" "CommonCrypto/CommonDigest.h" HAVE_MD5_UPDATE)
		check_symbol_exists("CC_SHA1_Update" "CommonCrypto/CommonDigest.h" HAVE_SHA1_UPDATE)
		check_symbol_exists("CC_SHA224_Update" "CommonCrypto/CommonDigest.h" HAVE_SHA224_UPDATE)
		check_symbol_exists("CC_SHA256_Update" "CommonCrypto/CommonDigest.h" HAVE_SHA256_UPDATE)
		check_symbol_exists("CC_SHA384_Update" "CommonCrypto/CommonDigest.h" HAVE_SHA384_UPDATE)
		check_symbol_exists("CC_SHA512_Update" "CommonCrypto/CommonDigest.h" HAVE_SHA512_UPDATE)
	endif()
endif()
if(NOT COMMONCRYPTO_FOUND)
	find_package(OpenSSL)

	if(OPENSSL_FOUND)
		set(HAVE_OPENSSL ON CACHE INTERNAL "OpenSSL is available")
		set(CRYPTO_INCLUDE_DIR "${OPENSSL_INCLUDE_DIR}" CACHE INTERNAL "crypto include directory")
		set(CRYPTO_LIBRARIES "${OPENSSL_CRYPTO_LIBRARY}" CACHE INTERNAL "crypto libraries to link")
		set(CMAKE_REQUIRED_INCLUDES "${CMAKE_REQUIRED_INCLUDES};${CRYPTO_INCLUDE_DIR}")
		set(CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES};${CRYPTO_LIBRARIES}")
		check_symbol_exists("MD5_Update" "openssl/md5.h" HAVE_MD5_UPDATE)
		check_symbol_exists("RIPEMD160_Update" "openssl/ripemd.h" HAVE_RIPEMD160_UPDATE)
		check_symbol_exists("SHA1_Update" "openssl/sha.h" HAVE_SHA1_UPDATE)
		check_symbol_exists("SHA224_Update" "openssl/sha.h" HAVE_SHA224_UPDATE)
		check_symbol_exists("SHA256_Update" "openssl/sha.h" HAVE_SHA256_UPDATE)
		check_symbol_exists("SHA384_Update" "openssl/sha.h" HAVE_SHA384_UPDATE)
		check_symbol_exists("SHA512_Update" "openssl/sha.h" HAVE_SHA512_UPDATE)
	endif()
endif()
cmake_pop_check_state()
if(NOT COMMONCRYPTO_FOUND AND NOT OPENSSL_FOUND)
	if(${ENABLE_MONETDB5} STREQUAL "YES")
		message(FATAL_ERROR "Neither OpenSSL or CommonCrypto library found, but required for MonetDB5")
	else()
		set(ENABLE_MONETDB5 "NO" CACHE STRING "Neither OpenSSL or CommonCrypto library found, but required for MonetDB5" FORCE)
		message(WARNING "Neither OpenSSL or CommonCrypto library found, but required for MonetDB5")
	endif()
endif()
if(NOT HAVE_${PASSWORD_BACKEND}_UPDATE)
	if(${ENABLE_MONETDB5} STREQUAL "YES")
		message(FATAL_ERROR "A crypto library was found, but checksum algorithm required for MonetDB5 was not found")
	else()
		set(HAVE_COMMONCRYPTO NO)
		set(OPENSSL_FOUND NO)
		set(ENABLE_MONETDB5 "NO" CACHE STRING "A crypto library was found, but checksum algorithm required for MonetDB5 was not found" FORCE)
		message(WARNING "A crypto library was found, but required checksum algorithm was not found, so disabling MonetDB5")
	endif()
endif()

# Regex library
if(NOT ${WITH_REGEX} MATCHES "^AUTO|PCRE|POSIX$")
	message(FATAL_ERROR "WITH_REGEX argument must be either AUTO, PCRE or POSIX")
endif()
if(NOT ${WITH_REGEX} STREQUAL "POSIX")
	include(FindPCRE)

	if(PCRE_FOUND)
		set(PCRE_REQUIRED_VERSION "4.5")
		if(PCRE_VERSION VERSION_LESS ${PCRE_REQUIRED_VERSION})
			if(${WITH_REGEX} STREQUAL "PCRE")
				message(FATAL_ERROR "PCRE library found, but the version is too old ${PCRE_VERSION} < ${PCRE_REQUIRED_VERSION}")
			else()
				message(WARNING "PCRE library found, but the version is too old ${PCRE_VERSION} < ${PCRE_REQUIRED_VERSION}")
			endif()
		else()
			cmake_push_check_state()
			set(CMAKE_REQUIRED_INCLUDES "${CMAKE_REQUIRED_INCLUDES};${PCRE_INCLUDE_DIR}")
			check_symbol_exists("PCRE_CONFIG_UTF8" "pcre.h" HAVE_LIBPCRE) #Test for UTF-8 support on PCRE library (PCRE_CONFIG_UTF8 is a macro)
			cmake_pop_check_state()
			if(NOT HAVE_LIBPCRE)
				if(${WITH_REGEX} STREQUAL "PCRE")
					message(FATAL_ERROR "PCRE library found but no UTF-8 support")
				else()
					message(STATUS "PCRE library found but no UTF-8 support")
				endif()
			endif()
		endif()
	elseif(${WITH_REGEX} STREQUAL "PCRE")
		message(FATAL_ERROR "PCRE library not found or not usable")
	else()
		message(WARNING "PCRE library not found or not usable")
	endif()
endif()
if(NOT HAVE_LIBPCRE)
	set(PCRE_INCLUDE_DIR "" CACHE INTERNAL "pcre include directory")
	set(PCRE_LIBRARIES "" CACHE INTERNAL "pcre libraries to link")
	check_symbol_exists("regcomp" "regex.h" HAVE_POSIX_REGEX)
endif()
if(NOT HAVE_LIBPCRE AND NOT HAVE_POSIX_REGEX)
	if(${ENABLE_MONETDB5} STREQUAL "YES")
		message(FATAL_ERROR "PCRE library or GNU regex library not found but required for MonetDB5")
	else()
		set(ENABLE_MONETDB5 "NO" CACHE STRING "PCRE library or GNU regex library not found but required for MonetDB5" FORCE)
		message(WARNING "PCRE library or GNU regex library not found but required for MonetDB5")
	endif()
endif()

# System libraries and optional extra libraries
set(DL_LIBRARIES "")
set(KVM_LIBRARIES "")
set(MATH_LIBRARIES "")
set(PSAPI_LIBRARIES "")
set(SOCKET_LIBRARIES "")
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
	set(DL_LIBRARIES "${CMAKE_DL_LIBS}")
endif()
if(${CMAKE_SYSTEM_NAME} MATCHES "^FreeBSD|DragonFly|NetBSD$") # Warning - I checked the man pages and only tested on FreeBSD yet
	set(KVM_LIBRARIES "kvm")
endif()
if(${CMAKE_SYSTEM_NAME} MATCHES "^Linux|FreeBSD|NetBSD$")
	set(MATH_LIBRARIES "m")
endif()
if(WIN32) # Both these libraries and respective include files (psapi.h and winsock2.h) come with the Windows SDK <version>, which should be installed with Visual Studio and set on the path by MSVC
	set(PSAPI_LIBRARIES "psapi") # We need the psapi library for GetProcessMemoryInfo function, which is no longer required from Windows 7 and Windows Server 2008 R2 up (the latter is suported until January 2020)
	set(SOCKET_LIBRARIES "ws2_32")
endif()
include(FindExtraLibraries)

# Header files
check_include_file("libintl.h" HAVE_LIBINTL_H) # We use it on Windows, and there are ports of this library there
check_include_file("pthread.h" HAVE_PTHREAD_H) # Distinguish between posix thread and win32 thread libraries

# C types existence
cmake_push_check_state()
set(CMAKE_EXTRA_INCLUDE_FILES "${CMAKE_EXTRA_INCLUDE_FILES};sys/types.h")
check_type_size(size_t SIZEOF_SIZE_T LANGUAGE C) #On C99, but we have to calculate the size
check_type_size(ssize_t SIZEOF_SSIZE_T LANGUAGE C)
if(NOT HAVE_SIZEOF_SSIZE_T) # Set a default value
	if(CMAKE_SIZEOF_VOID_P EQUAL 8)
		set(ssize_t "int64_t")
	else()
		set(ssize_t "int32_t")
	endif()
	set(SIZEOF_SSIZE_T ${CMAKE_SIZEOF_VOID_P})
endif()
check_type_size(char SIZEOF_CHAR LANGUAGE C)
check_type_size(short SIZEOF_SHORT LANGUAGE C)
check_type_size(int SIZEOF_INT LANGUAGE C)
check_type_size(long SIZEOF_LONG LANGUAGE C)
set(SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
check_type_size(wchar_t SIZEOF_WCHAR_T LANGUAGE C)
if(NOT ${ENABLE_INT128} STREQUAL "NO")
	check_type_size(__int128 SIZEOF___INT128 LANGUAGE C)
	check_type_size(__int128_t SIZEOF___INT128_T LANGUAGE C)
	check_type_size(__uint128_t SIZEOF___UINT128_T LANGUAGE C)
	if(HAVE_SIZEOF___INT128 OR HAVE_SIZEOF___INT128_T OR HAVE_SIZEOF___UINT128_T)
		set(HAVE_HGE ON CACHE INTERNAL "Huge integers are available")
	elseif(${ENABLE_INT128} STREQUAL "YES")
		message(FATAL_ERROR "128-bit integers not supported by this compiler")
	else()
		set(ENABLE_INT128 "NO" CACHE STRING "128-bit integers not supported by this compiler" FORCE)
		message(STATUS "128-bit integers not supported by this compiler")
	endif()
endif()
cmake_pop_check_state()

# Non portable functions
check_symbol_exists("accept4" "sys/types.h;sys/socket.h" HAVE_ACCEPT4) # Some libc versions on Linux distributions don't have it
check_symbol_exists("fallocate" "fcntl.h" HAVE_FALLOCATE) # Linux specific, in the future, it might be ported to other platforms
check_symbol_exists("madvise" "sys/mman.h" HAVE_MADVISE) # The Linux kernel can be compiled without madvise
check_symbol_exists("mremap" "sys/mman.h" HAVE_MREMAP) # Linux specific, in the future, it might be ported to other platforms
check_symbol_exists("pipe2" "fcntl.h;unistd.h" HAVE_PIPE2) # Some libc versions on Linux distributions don't have it
check_symbol_exists("posix_fallocate" "fcntl.h" HAVE_POSIX_FALLOCATE) # Some POSIX systems don't have it (e.g. Macos)
check_symbol_exists("posix_madvise" "sys/mman.h" HAVE_POSIX_MADVISE) # The Linux kernel can be compiled without madvise, and posix_madvise is implemented using madvise
check_symbol_exists("semtimedop" "sys/types.h;sys/ipc.h;sys/sem.h" HAVE_SEMTIMEDOP) # Some libc versions on Linux distributions don't have it

if(${ENABLE_GDK} STREQUAL "NO")
	if(${ENABLE_MONETDB5} STREQUAL "YES")
		message(FATAL_ERROR "MonetDB5 requires GDK")
	elseif(${ENABLE_MONETDB5} STREQUAL "AUTO")
		set(ENABLE_MONETDB5 "NO" CACHE STRING "MonetDB5 requires GDK" FORCE)
	endif()
else()
	set(HAVE_GDK ON CACHE INTERNAL "GDK layer is available")
endif()
if(${ENABLE_MONETDB5} STREQUAL "NO")
	if(${ENABLE_SQL} STREQUAL "YES")
		message(FATAL_ERROR "MonetDB/SQL requires MonetDB5")
	elseif(${ENABLE_SQL} STREQUAL "AUTO")
		set(ENABLE_SQL "NO" CACHE STRING "MonetDB/SQL requires MonetDB5" FORCE)
	endif()
else()
	set(HAVE_MONETDB5 YES)
endif()
if(NOT ${ENABLE_SQL} STREQUAL "NO") # The SQL module requires the monetdb5 module, so monetdb5 should compile first
	set(HAVE_SQL ON CACHE INTERNAL "MonetDB/SQL is available")
endif()
if(NOT ${ENABLE_EMBEDDED} STREQUAL "NO")
	set(HAVE_EMBEDDED ON CACHE INTERNAL "Embedded MonetDB is enabled")
endif()
if(NOT ${ENABLE_CONSOLE} STREQUAL "NO")
	set(HAVE_CONSOLE ON CACHE INTERNAL "Direct console on the server is available")
endif()
if(NOT ${ENABLE_STATIC_ANALYSIS} STREQUAL "NO")
	set(STATIC_CODE_ANALYSIS ON CACHE INTERNAL "Static code analysis is available")
endif()
if("${ENABLE_TESTING}" STREQUAL "YES" AND NOT HAVE_PYTHON3)
	message(FATAL_ERROR "A valid Python 3 interpreter is required for MonetDB testing tools")
endif()
if(NOT "${ENABLE_TESTING}" STREQUAL "NO" AND HAVE_PYTHON3)
	set(HAVE_TESTING ON CACHE INTERNAL "Test suite is enabled")
endif()

# Very Important, check for compiler flags after all libraries checks, otherwise extra compiler flags might break the
# library tests
function(MT_removeCompilerFlag REGEX_EXPRESSION FLAG_TO_REMOVE OLD_FLAGS BUILD NEW_FLAGS)
	if("${OLD_FLAGS}" MATCHES "${REGEX_EXPRESSION}")
		string(REGEX REPLACE "${REGEX_EXPRESSION}" "" NEW_VAR "${OLD_FLAGS}")
		set(${NEW_FLAGS} "${NEW_VAR}" PARENT_SCOPE)
		message(STATUS "Removed ${FLAG_TO_REMOVE} flag for ${BUILD} build(s)")
	endif()
endfunction()

if(MSVC)
	MT_addCompilerFlag("/GF" "/GF" "${CMAKE_C_FLAGS}" "all" CMAKE_C_FLAGS)
	MT_addCompilerFlag("/nologo" "/nologo" "${CMAKE_C_FLAGS}" "all" CMAKE_C_FLAGS)
	MT_addCompilerFlag("/Oi" "/Oi" "${CMAKE_C_FLAGS}" "all" CMAKE_C_FLAGS)
	# Remove all /Ox but /Oi on all, /Od on Debug and /Ox on others
	MT_removeCompilerFlag("/O[0-9a-ce-hj-zA-CE-HJ-Z]+[ ]?" "/Ox" "${CMAKE_C_FLAGS_DEBUG}" "Debug" CMAKE_C_FLAGS_DEBUG)
	MT_removeCompilerFlag("/O[0-9a-hj-wyzA-HJ-WYZ]+[ ]?" "/Ox" "${CMAKE_C_FLAGS_RELEASE}" "Release" CMAKE_C_FLAGS_RELEASE)
	MT_removeCompilerFlag("/O[0-9a-hj-wyzA-HJ-WYZ]+[ ]?" "/Ox" "${CMAKE_C_FLAGS_RELWITHDEBINFO}" "RelWithDebugInfo" CMAKE_C_FLAGS_RELWITHDEBINFO)
	MT_removeCompilerFlag("/O[0-9a-rt-wyzA-RT-WYZ]+[ ]?" "/Ox" "${CMAKE_C_FLAGS_MINSIZEREL}" "MinSizeRel" CMAKE_C_FLAGS_MINSIZEREL)
	MT_addCompilerFlag("/Od" "/Od" "${CMAKE_C_FLAGS_DEBUG}" "Debug" CMAKE_C_FLAGS_DEBUG)
	MT_addCompilerFlag("/Ox" "/Ox" "${CMAKE_C_FLAGS_RELEASE}" "Release" CMAKE_C_FLAGS_RELEASE)
	MT_addCompilerFlag("/Ox" "/Ox" "${CMAKE_C_FLAGS_RELWITHDEBINFO}" "RelWithDebugInfo" CMAKE_C_FLAGS_RELWITHDEBINFO)
	MT_addCompilerFlag("/Os" "/Os" "${CMAKE_C_FLAGS_MINSIZEREL}" "MinSizeRel" CMAKE_C_FLAGS_MINSIZEREL)
elseif(${CMAKE_C_COMPILER_ID} STREQUAL "Intel")
	MT_removeCompilerFlag("${COMPILER_OPTION}O[0-9]+[ ]?" "${COMPILER_OPTION}Ox" "${CMAKE_C_FLAGS_DEBUG}" "Debug" CMAKE_C_FLAGS_DEBUG)
	MT_removeCompilerFlag("${COMPILER_OPTION}O[0-24-9]+[ ]?" "${COMPILER_OPTION}Ox" "${CMAKE_C_FLAGS_RELEASE}" "Release" CMAKE_C_FLAGS_RELEASE)
	MT_removeCompilerFlag("${COMPILER_OPTION}O[0-24-9]+[ ]?" "${COMPILER_OPTION}Ox" "${CMAKE_C_FLAGS_RELWITHDEBINFO}" "RelWithDebugInfo" CMAKE_C_FLAGS_RELWITHDEBINFO)
	MT_removeCompilerFlag("${COMPILER_OPTION}O[02-9]+[ ]?" "${COMPILER_OPTION}Ox" "${CMAKE_C_FLAGS_MINSIZEREL}" "MinSizeRel" CMAKE_C_FLAGS_MINSIZEREL)
	if(WIN32)
		MT_addCompilerFlag("${COMPILER_OPTION}Od" "${COMPILER_OPTION}Od" "${CMAKE_C_FLAGS_DEBUG}" "Debug" CMAKE_C_FLAGS_DEBUG)
	endif()
	MT_addCompilerFlag("${COMPILER_OPTION}O3" "${COMPILER_OPTION}O3" "${CMAKE_C_FLAGS_RELEASE}" "Release" CMAKE_C_FLAGS_RELEASE)
	MT_addCompilerFlag("${COMPILER_OPTION}O3" "${COMPILER_OPTION}O3" "${CMAKE_C_FLAGS_RELWITHDEBINFO}" "RelWithDebugInfo" CMAKE_C_FLAGS_RELWITHDEBINFO)
	MT_addCompilerFlag("${COMPILER_OPTION}O1" "${COMPILER_OPTION}O1" "${CMAKE_C_FLAGS_MINSIZEREL}" "MinSizeRel" CMAKE_C_FLAGS_MINSIZEREL)
	if(CMAKE_VERSION VERSION_LESS 3.1)
		MT_addCompilerFlag("${COMPILER_OPTION}${INTEL_OPTION_EXTRA}std=c99" "${COMPILER_OPTION}${INTEL_OPTION_EXTRA}std=c99" "${CMAKE_C_FLAGS}" "all" CMAKE_C_FLAGS)
	endif()
else() # GCC, Clang and Apple clang
	# Remove existing -Ox flags
	MT_removeCompilerFlag("-O[1-2sS]+[ ]?" "-Ox" "${CMAKE_C_FLAGS_DEBUG}" "Debug" CMAKE_C_FLAGS_DEBUG)
	MT_removeCompilerFlag("-O[0-2sS]+[ ]?" "-Ox" "${CMAKE_C_FLAGS_RELEASE}" "Release" CMAKE_C_FLAGS_RELEASE)
	MT_removeCompilerFlag("-O[0-2sS]+[ ]?" "-Ox" "${CMAKE_C_FLAGS_RELWITHDEBINFO}" "RelWithDebugInfo" CMAKE_C_FLAGS_RELWITHDEBINFO)
	MT_removeCompilerFlag("-O[0-3]+[ ]?" "-Ox" "${CMAKE_C_FLAGS_MINSIZEREL}" "MinSizeRel" CMAKE_C_FLAGS_MINSIZEREL)
	MT_addCompilerFlag("-O0" "-O0" "${CMAKE_C_FLAGS_DEBUG}" "Debug" CMAKE_C_FLAGS_DEBUG)
	MT_addCompilerFlag("-O3" "-O3" "${CMAKE_C_FLAGS_RELWITHDEBINFO}" "RelWithDebugInfo" CMAKE_C_FLAGS_RELWITHDEBINFO)
	MT_addCompilerFlag("-O3" "-O3" "${CMAKE_C_FLAGS_RELEASE}" "Release" CMAKE_C_FLAGS_RELEASE)
	MT_addCompilerFlag("-Os" "-Os" "${CMAKE_C_FLAGS_MINSIZEREL}" "MinSizeRel" CMAKE_C_FLAGS_MINSIZEREL)
	# Replace -g flag with -g3
	MT_removeCompilerFlag("-g([ ]|$)" "-g" "${CMAKE_C_FLAGS_DEBUG}" "Debug" CMAKE_C_FLAGS_DEBUG)
	MT_removeCompilerFlag("-g([ ]|$)" "-g" "${CMAKE_C_FLAGS_RELWITHDEBINFO}" "RelWithDebugInfo" CMAKE_C_FLAGS_RELWITHDEBINFO)
	MT_addCompilerFlag("-g3" "-g3" "${CMAKE_C_FLAGS_DEBUG}" "Debug" CMAKE_C_FLAGS_DEBUG)
	MT_addCompilerFlag("-g3" "-g3" "${CMAKE_C_FLAGS_RELWITHDEBINFO}" "RelWithDebugInfo" CMAKE_C_FLAGS_RELWITHDEBINFO)

	if(NOT ${CMAKE_C_FLAGS_RELEASE} MATCHES "D_FORTIFY_SOURCE")
		set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -D_FORTIFY_SOURCE=2")
		set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -D_FORTIFY_SOURCE=2")
		set(CMAKE_C_FLAGS_MINSIZEREL "${CMAKE_C_FLAGS_MINSIZEREL} -D_FORTIFY_SOURCE=2")
		message(STATUS "Added definition _FORTIFY_SOURCE=2 for Release, RelWithDebugInfo and MinSizeRel builds")
	endif()
	# Add -fomit-frame-pointer for non debug builds
	MT_addCompilerFlag("-fomit-frame-pointer" "-fomit-frame-pointer" "${CMAKE_C_FLAGS_RELEASE}" "Release" CMAKE_C_FLAGS_RELEASE)
	MT_addCompilerFlag("-fomit-frame-pointer" "-fomit-frame-pointer" "${CMAKE_C_FLAGS_MINSIZEREL}" "MinSizeRel" CMAKE_C_FLAGS_MINSIZEREL)
	if(CMAKE_VERSION VERSION_LESS 3.1) # Set C99
		MT_addCompilerFlag("-std=c99" "-std=c99" "${CMAKE_C_FLAGS}" "all" CMAKE_C_FLAGS)
	endif()

	# Linker flags
	if(${CMAKE_C_COMPILER_ID} STREQUAL "AppleClang")
		MT_addCompilerFlag("-Wl,-undefined,dynamic_lookup" "-Wl,-undefined,dynamic_lookup" "${CMAKE_SHARED_LINKER_FLAGS}" "all shared libraries" CMAKE_SHARED_LINKER_FLAGS)
		MT_addCompilerFlag("-Wl,-undefined,dynamic_lookup" "-Wl,-undefined,dynamic_lookup" "${CMAKE_MODULE_LINKER_FLAGS}" "all shared library modules" CMAKE_MODULE_LINKER_FLAGS)
		MT_addCompilerFlag("-Wl,-single_module" "-Wl,-single_module" "${LINK_OPTIONS}" "all" LINK_OPTIONS)
	else()
		MT_addCompilerFlag("-Wl,-Bsymbolic-functions" "-Wl,-Bsymbolic-functions" "${LINK_OPTIONS}" "all" LINK_OPTIONS)
		MT_addCompilerFlag("-Wl,--export-dynamic" "-Wl,--export-dynamic" "${LINK_OPTIONS}" "all" LINK_OPTIONS)
	endif()
endif()

function(MT_checkCompilerFlag Flag) # WARNING this doesn't work with LINK_OPTIONS yet!
	if(NOT "${CMAKE_C_FLAGS}" MATCHES "${Flag}")
		string(REGEX REPLACE "[-/=,]" "" FLAG_TEST "${Flag}") # cmake will get confused if we don't remove - / = ,
		cmake_push_check_state()
		set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${Flag}")
		check_c_source_compiles("int main(int argc,char** argv){(void)argc;(void)argv;return 0;}" COMPILER_${FLAG_TEST})
		cmake_pop_check_state()
		if(COMPILER_${FLAG_TEST})
			set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Flag}" PARENT_SCOPE)
		endif()
	endif()
endfunction()

if(${ENABLE_STRICT} STREQUAL "YES")
	if(${CMAKE_C_COMPILER_ID} MATCHES "^GNU|Clang|AppleClang$")
		MT_checkCompilerFlag("-W")
		MT_checkCompilerFlag("-Werror")
		MT_checkCompilerFlag("-Wall")
		MT_checkCompilerFlag("-Wextra")
		MT_checkCompilerFlag("-Werror-implicit-function-declaration")
		MT_checkCompilerFlag("-Wpointer-arith")
		MT_checkCompilerFlag("-Wundef")
		MT_checkCompilerFlag("-Wformat=2")
		MT_checkCompilerFlag("-Wformat-overflow=1")
		MT_checkCompilerFlag("-Wno-format-truncation")
		MT_checkCompilerFlag("-Wno-format-nonliteral")
		MT_checkCompilerFlag("-Winit-self")
		MT_checkCompilerFlag("-Winvalid-pch")
		MT_checkCompilerFlag("-Wmissing-declarations")
		MT_checkCompilerFlag("-Wmissing-format-attribute")
		MT_checkCompilerFlag("-Wmissing-prototypes")
		MT_checkCompilerFlag("-Wold-style-definition")
		MT_checkCompilerFlag("-Wpacked")
		MT_checkCompilerFlag("-Wunknown-pragmas")
		MT_checkCompilerFlag("-Wvariadic-macros")
		MT_checkCompilerFlag("-fstack-protector-all")
		MT_checkCompilerFlag("-Wstack-protector")
		MT_checkCompilerFlag("-Wpacked-bitfield-compat")
		MT_checkCompilerFlag("-Wsync-nand")
		MT_checkCompilerFlag("-Wjump-misses-init")
		MT_checkCompilerFlag("-Wmissing-include-dirs")
		MT_checkCompilerFlag("-Wlogical-op")
		MT_checkCompilerFlag("-Wduplicated-cond")
		MT_checkCompilerFlag("-Wduplicated-branches")
		MT_checkCompilerFlag("-Wrestrict")
		MT_checkCompilerFlag("-Wnested-externs")
		MT_checkCompilerFlag("-Wunreachable-code")
	elseif(${CMAKE_C_COMPILER_ID} STREQUAL "Intel")
		if(NOT WIN32)
			MT_checkCompilerFlag("${COMPILER_OPTION}shared-intel")
		endif()
		MT_checkCompilerFlag("${COMPILER_OPTION}Wall")
		MT_checkCompilerFlag("${COMPILER_OPTION}Wcheck")
		MT_checkCompilerFlag("${COMPILER_OPTION}Werror-all")
		MT_checkCompilerFlag("${COMPILER_OPTION}${INTEL_OPTION_EXTRA}we140,147,181,266,271,593,810")
		MT_checkCompilerFlag("${COMPILER_OPTION}${INTEL_OPTION_EXTRA}wd193,279,981,1357,1418,1419,2259")
	elseif(MSVC)
		MT_addCompilerFlag("/WX" "/WX" "${CMAKE_C_FLAGS}" "all" CMAKE_C_FLAGS)
	endif()
endif()

if(NOT ${ENABLE_SANITIZER} STREQUAL "NO")
	if(${CMAKE_C_COMPILER_ID} STREQUAL "GNU")
		MT_checkCompilerFlag("-fsanitize=address")
		MT_checkCompilerFlag("-fno-omit-frame-pointer")
		add_definitions(-DNO_ATOMIC_INSTRUCTIONS)
	else()
		message(FATAL_ERROR "Sanitizer only supported with GCC")
	endif()
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "Compilation flags for all builds" FORCE)
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}" CACHE STRING "Debug flags" FORCE)
set(CMAKE_C_FLAGS_MINSIZEREL "${CMAKE_C_FLAGS_MINSIZEREL}" CACHE STRING "MinSizeRel flags" FORCE)
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}" CACHE STRING "Release flags" FORCE)
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}" CACHE STRING "RelWithDebugInfo flags" FORCE)
set(CMAKE_C_LINK_FLAGS "${LINK_OPTIONS}" CACHE STRING "Link flags for all builds" FORCE) # This variable is depreacted but still used
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS}" CACHE STRING "Shared library module link options for all builds" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}" CACHE STRING "Shared library link options for all builds" FORCE)
set(LINK_OPTIONS "${LINK_OPTIONS}" CACHE STRING "Link options for all builds" FORCE)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_subdirectory(buildtools)
add_subdirectory(common)
add_subdirectory(clients) # Must come before monetdb5!
if(HAVE_GDK)
	add_subdirectory(gdk)
endif()
if(HAVE_MONETDB5)
	add_subdirectory(monetdb5)
	add_subdirectory(geom) # The SHP vault requires the geom module, so the geom module should compile before SQL
endif()
if(HAVE_SQL)
	add_subdirectory(sql)
endif()
add_subdirectory(tools)
if(HAVE_TESTING)
	add_subdirectory(testing) # This one must be the last visted directory!
endif()

# Set monetdb_config.h at the end!
configure_file(monetdb_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/monetdb_config.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/monetdb_config.h DESTINATION ${INCLUDEDIR}/monetdb)

# Cpack configuration variables
include(Distribute)
