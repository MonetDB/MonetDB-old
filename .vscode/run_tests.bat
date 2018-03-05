rem call environment variables for Monetdb compilation and runtime
call %~dp0/monetdb-environment.bat

cd %SOURCE%\sql\test\configurable_rss\Tests

Mtest.py
