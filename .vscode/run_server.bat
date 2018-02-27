rem call environment variables for Monetdb compilation and runtime
call %~dp0/monetdb-environment.bat

rem Start the Monetdb5 server
%PREFIX%\M5server.bat
