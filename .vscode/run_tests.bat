rem call environment variables for Monetdb compilation and runtime
call %~dp0/monetdb-environment.bat

Mtest.py -r
