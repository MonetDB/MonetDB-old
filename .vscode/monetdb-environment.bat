rem Set build variables for monetdb
set SOURCE=%USERPROFILE%\Sources\monetdb
set BUILD=%SOURCE%\build
set PREFIX=%USERPROFILE%\monetdb-installation
set Path=%PREFIX%\bin;%PREFIX%\lib;%Path%

rem additional libraries and additional tools

rem Look in Cygwin's bin as a default location for programs, e.g. Bison.
rem When appropriate, Windows specific packages like the Python distributions are given priority over the possible Cygwin alternatives.
set CYGWIN=C:\cygwin64
set Path=%CYGWIN%\bin;%Path%

rem PCRE
set LIBPCRE=%ProgramFiles%\PCRE
set Path=%LIBPCRE%\bin;%Path%
set Path=%LIBPCRE%\lib;%Path%

rem OpenSSL
set OPENSSLDIR=C:\openssl
set Path=%OPENSSLDIR%\bin;%Path%
set Path=%OPENSSLDIR%\lib;%Path%

rem libXML
set LIBXML2=C:\libxml2-2.9.3-win32-x86_64
set Path=%LIBXML2%\bin;%Path%
set Path=%LIBXML2%\lib;%Path%

rem zlib
set LIBZLIB=C:\zlib-1.2.11-win64-x86_64
set Path=%LIBZLIB%\bin;%Path%
set Path=%LIBZLIB%\lib;%Path%

rem bzip2
set LIBBZ2=C:\bzip2-1.0.6
set Path=%LIBBZ2%\bin;%Path%
set Path=%LIBBZ2%\lib;%Path%

rem iconv
set LIBICONV=C:\iconv-1.14-win32-x86_64
set Path=%LIBICONV%\bin;%Path%
set Path=%LIBICONV%\lib;%Path%

rem perl
set PERLDIR=C:\Perl64
set Path=%PERLDIR%\bin;%Path%
set Path=%PERLDIR%\lib;%Path%

rem geos
set GEOSDIR=C:\geos-3.4.2
set Path=%GEOSDIR%\bin;%Path%
set Path=%GEOSDIR%\lib;%Path%

rem python3
set PYTHON3=C:\Python36
set Path=%PYTHON3%;%Path%
set PYTHON3_LIBDIR=%PYTHON3%\lib

rem python2
set PYTHON2=C:\Python27
set Path=%PYTHON2%;%Path%
set PYTHON2_LIBDIR=%PYTHON2%\lib

rem Python module search path
set PYTHONPATH=%PREFIX%\lib\site-packages;%PYTHONPATH%
