Cmake 3.1 is required on Windows, On all other platforms, cmake 3.0 suffices.
On CentOS 6, enable epel repository, then install cmake3 package.
While running the cmake command set internal variables to change the build properties in the form -D<var>=<value>

- Variables intrinsic to Cmake that we set on our scripts
CMAKE_BUILD_TYPE - Set the build type: Debug or Release (default Debug)
CMAKE_C_FLAGS - C compilation flags used for all builds
CMAKE_C_FLAGS_DEBUG - C compilation flags used for Debug build
CMAKE_C_FLAGS_MINSIZEREL - C compilation flags used for Release build woth minimal size
CMAKE_C_FLAGS_RELEASE - C compilation flags used for Release build
CMAKE_C_FLAGS_RELWITHDEBINFO - C compilation flags used for Release build with debug symbols
CMAKE_C_LINK_FLAGS - Linker options (deprecated for LINK_OPTIONS variable, but still used by some Cmake modules)
CMAKE_MODULE_LINKER_FLAGS - Linker options for shared library modules
CMAKE_MODULE_PATH - Location of custom CMake modules (in cmake directory)
CMAKE_SHARED_LINKER_FLAGS - Linker options for shared libraries
LINK_OPTIONS - Linker options for translation units for all builds

- We set the library prefix variables on Windows compilation because we don't follow the naming convention there :(
CMAKE_IMPORT_LIBRARY_PREFIX
CMAKE_SHARED_LIBRARY_PREFIX
CMAKE_SHARED_MODULE_PREFIX
CMAKE_STATIC_LIBRARY_PREFIX

- These other Cmake variables worth to set
CMAKE_C_COMPILER_ID - Which compiler to use
CMAKE_INSTALL_PREFIX - Installation directory

- Compilation options
ENABLE_SANITIZER
ENABLE_STATIC_ANALYSIS
ENABLE_STRICT

- Available MonetDB features
ENABLE_CINTEGRATION
ENABLE_EMBEDDED - TODO check this
ENABLE_FITS
ENABLE_GDK
ENABLE_GEOM
ENABLE_INT128
ENABLE_LIDAR
ENABLE_MAPI
ENABLE_MONETDB5
ENABLE_NETCDF
ENABLE_ODBC
ENABLE_PY3INTEGRATION
ENABLE_RINTEGRATION
ENABLE_SAMTOOLS
ENABLE_SANITIZER
ENABLE_SHP
ENABLE_SQL
ENABLE_STATIC_ANALYSIS
ENABLE_STRICT
ENABLE_TESTING

- GNU installation directories (only on UNIX platforms)
LOGDIR
RUNDIR

PASSWORD_BACKEND

- Extra libraries
WITH_BZ2
WITH_CURL
WITH_LIBLZMA
WITH_LIBXML2
WITH_LZ4
WITH_PROJ
WITH_READLINE
WITH_REGEX
WITH_SNAPPY
WITH_UUID
WITH_VALGRIND
WITH_ZLIB

Linux notes:
 - libtools files are not generated yet.

We use CPack module to generate source tar files with compression. After cmake run:
 cpack -G <generator> --config <path to compilation directory where CPackConfig.cmake is located>
For generators we use TBZ2 TGZ TXZ and ZIP. Check with 'cpack --help' for details.
