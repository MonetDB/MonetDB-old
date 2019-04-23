While running the cmake command set internal variables to change the build properties in the form -D<var>=<value>

- Variables intrinsic to Cmake
CMAKE_BUILD_TYPE - Set the build type: Debug or Release (default Debug)
CMAKE_C_FLAGS - C compilation flags used for all builds
CMAKE_C_FLAGS_DEBUG - C compilation flags used for Debug build
CMAKE_C_FLAGS_RELEASE - C compilation flags used for Release build
CMAKE_INSTALL_PREFIX - Installation directory
CMAKE_C_LINK_FLAGS - Linker options for all builds

- Compilation options
ENABLE_ASSERT
ENABLE_DEVELOPER
ENABLE_SANITIZER
ENABLE_STATIC_ANALYSIS
ENABLE_STRICT

- Available Monetdb features
ENABLE_EMBEDDED
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
ENABLE_SHP
ENABLE_SQL
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
