While running the cmake command set internal variables to change the build properties in the form -D<var>=<value>

- Variables intrinsic to Cmake
CMAKE_BUILD_TYPE - Set the build type: Debug or Release (default Debug)
CMAKE_C_FLAGS - C compilation flags used for all builds
CMAKE_C_FLAGS_DEBUG - C compilation flags used for Debug build
CMAKE_C_FLAGS_RELEASE - C compilation flags used for Release build
CMAKE_INSTALL_PREFIX - Installation directory
LINK_OPTIONS - Linker options for all builds

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
ENABLE_PY2INTEGRATION
ENABLE_PY3INTEGRATION
ENABLE_RINTEGRATION
ENABLE_SHP
ENABLE_SQL
ENABLE_TESTING

- GNU installation directories (only on UNIX platforms)
LOGDIR
RUNDIR

PASSWORD_BACKEND

- Python configuration
PYTHON2
PYTHON2_CONFIG
PYTHON2_LIBDIR
PYTHON3
PYTHON3_CONFIG
PYTHON3_LIBDIR

- Extra libraries
WITH_BZ2
WITH_CURL
WITH_GDAL
WITH_GEOS
WITH_LIBLAS
WITH_LIBLZMA
WITH_LIBXML2
WITH_LZ4
WITH_OPENSSL (not available on MacOS X)
WITH_PROJ
WITH_PTHREAD (not available on Windows)
WITH_READLINE
WITH_REGEX
WITH_SAMTOOLS
WITH_SNAPPY
WITH_UNIXODBC (only if ENABLE_ODBC enabled)
WITH_UUID
WITH_VALGRIND
WITH_ZLIB

Linux notes:
 - Make install doesn't run ldconfig, use LD_LIBRARY_PATH https://cmake.org/pipermail/cmake/2016-June/063721.html
 - libtools files are not generated yet.
