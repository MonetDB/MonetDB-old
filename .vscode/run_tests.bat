rem call environment variables for Monetdb compilation and runtime
call %~dp0/monetdb-environment.bat

echo %PATH%

Mtest.py -r
