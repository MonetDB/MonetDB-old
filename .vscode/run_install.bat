rem call environment variables for Monetdb compilation and runtime
call %~dp0/monetdb-environment.bat

rem Set the environment variables for Visual Studio 2017 native x64.
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

cd %BUILD%

nmake -nologo -f ..\NT\Makefile bits=%BITS% prefix=%PREFIX% install
