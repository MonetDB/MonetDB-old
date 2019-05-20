cmake 3.7 is required on Windows (FindOpenSSL script), while on UNIX platforms cmake 3.0 suffices.
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

Available build types:
- Debug (default, use -DCMAKE_BUILD_TYPE=... to override)
- Release
- RelWithDebugInfo
- MinSizeRel (Release optimized for size)

- We set the library prefix variables on Windows compilation because we don't follow the naming convention there :(
CMAKE_IMPORT_LIBRARY_PREFIX
CMAKE_SHARED_LIBRARY_PREFIX
CMAKE_SHARED_MODULE_PREFIX
CMAKE_STATIC_LIBRARY_PREFIX

- These other Cmake variables worth to set
CMAKE_C_COMPILER_ID - Which compiler to use
CMAKE_INSTALL_PREFIX - Installation directory

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

- Digest algorithm to use for mserver authentication messages
PASSWORD_BACKEND - MD5, SHA1, RIPEMD160, SHA224, SHA256, SHA384 or SHA512, defaults to SHA512

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

What I tested so far:
- Linux with GCC 4.4.7 and Clang 3.4.2 with ENABLE_STRICT=NO. For newer versions of Clang and GCC I compiled
successfully with restrict flags. Tested on distributions Debian 8, Ubuntu 16.04, CentOS 6.10 and Fedora 30.
Also tested with Intel C++ Compiler 19. I generated both GNU make and ninja build files.
- FreeBSD 12 with GCC 8.3.0 and Clang 6.0.1.
- MacOS 10.13 with Apple Clang 10.
- Windows with Visual Studio 2015 (earliest version with enough C99 support). I tried "v140_clang_c2" toolset, but the
compiler crashed during compilation. I guess it's because of an earlier version of Clang (3.8). I tried the "LLVM_v140"
toolset with Clang 9.0.0 and compiled successfully with no restrictions. Some of the warnings triggered with
restrictions are compiler bugs, others we should ourselves. I didn't go further because we don't officially support
Clang on Windows. Also tested with Intel C++ Compiler 18 using "Intel C++ Compiler 18.0" toolset inside Cygwin. I
generated Visual Studio project files for every compiler plus nmake files and jom nmake files (for parallel builds) for
MSVC.

I haven't built 32-bit binaries yet.
What I did NOT test (we don't support):
- Solaris
- Any other Unixes: AIX, HP-UX, IRIX, Minix...
- Cygwin
- MinGW and MinGW-w64 compilers on Windows

Note that the install task depends on the build, so the build task will be executed if not so during the installation.
During generation phase, by default the current directory will be used to generate build files. Another directory can be
specified with the -B parameter.
------------------------------------------------------------------------------------------------------------------------
--UNIX with default compiler (first found on PATH) and UNIX makefiles:
cmake -DCMAKE_INSTALL_PREFIX=<installation dir> <source dir>
make -j<number of parallel builds>
make -j<number of parallel builds> install

------------------------------------------------------------------------------------------------------------------------
--UNIX with ninja generator and a compiler other than the default one:
cmake -G Ninja -DCMAKE_C_COMPILER=<path to compiler> -DCMAKE_INSTALL_PREFIX=<installation dir> <source dir>
ninja -j<number of parallel builds>
ninja -j<number of parallel builds> install

------------------------------------------------------------------------------------------------------------------------
--Xcode project
cmake -G Xcode -DCMAKE_INSTALL_PREFIX=<installation dir> <source dir>
xcodebuild
cmake --build <compile dir> --target install

------------------------------------------------------------------------------------------------------------------------
--Visual Studio 2015 project using MSVC:
cmake -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=<installation dir> <source dir>
cmake --build <compile dir> --target ALL_BUILD
cmake --build <compile dir> --target INSTALL

------------------------------------------------------------------------------------------------------------------------
--Visual Studio 2015 project with Intel compiler toolset that I used:
cmake -G "Visual Studio 14 2015 Win64" -T "Intel C++ Compiler 18.0" -DCMAKE_INSTALL_PREFIX=<installation dir> <source dir>
cmake --build <compile dir> --target ALL_BUILD
cmake --build <compile dir> --target INSTALL

------------------------------------------------------------------------------------------------------------------------
--Visual Studio 2015 project with LLVM toolset that I used:
cmake -G "Visual Studio 14 2015 Win64" -T "LLVM_v140" -DCMAKE_INSTALL_PREFIX=<installation dir> -DENABLE_STRICT=NO <source dir>
cmake --build <compile dir> --target ALL_BUILD
cmake --build <compile dir> --target INSTALL

------------------------------------------------------------------------------------------------------------------------
--nmake (no Visual Studio files are generated, thus the generation is faster)
--The vcvarsall script must run first to set the path and environment variables for 64-bit build:
vcvarsall amd64
cmake -G "NMake Makefiles" -DCMAKE_INSTALL_PREFIX=<installation dir> <source dir>
cmake --build <compile dir> --target all
cmake --build <compile dir> --target install

------------------------------------------------------------------------------------------------------------------------
--JOM is a adaptation of nmake from Qt to support parallel builds: https://wiki.qt.io/Jom
--The vcvarsall script must run first:
vcvarsall amd64
cmake -G "NMake Makefiles JOM" -DCMAKE_INSTALL_PREFIX=<installation dir> <source dir>
cmake --build <compile dir> --target all -- -j<number of parallel builds>
cmake --build <compile dir> --target install

------------------------------------------------------------------------------------------------------------------------
--We use CPack module to generate source tar files with compression. After cmake run:
--For generators we use TBZ2 TGZ TXZ and ZIP. Check with 'cpack --help' for details.
cpack -G <generator> --config <path to compilation directory where CPackConfig.cmake file is located>
