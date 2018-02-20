rem call environment variables for Monetdb compilation and runtime
call %~dp0/monetdb-environment.bat

rem Set the environment variables for Visual Studio 2017 native x64.
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

cd %BUILD%

nmake -nologo -f ..\NT\Makefile "prefix=%PREFIX%" HAVE_LIBBZ2=1 HAVE_MONETDB5=1 HAVE_SQL=1 HAVE_PCRE=1 HAVE_GDK=1 HAVE_OPENSSL=1 HAVE_MAPI=1 HAVE_GEOM=1 HAVE_PYTHON=1 HAVE_TESTING=1 "MAKE_INCLUDEFILE=%SOURCE%\.vscode\extraMakefile" install
