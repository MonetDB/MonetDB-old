# Building
The embedded Java version of MonetDB requires a few options to be set in order to build the required libraries and NOT build some of the undeeded binaries.

## Setting environmental variables
In order for the embedded Java to find the MonetDB libraries we are currently using an environmental variables. The same one would be used as a prefix when configuring the MonetDB build (see blow in 'Building MonetDB).
$ export MONETDB_HOME=<location to installation directory>

## Building MonetDB
Navigate the the main sources dir and bootstrap the build.
$ sh bootsrap
Create a BUILD directory and run the configure script found in the main sources dir. Set a prefix location using 
$ ../configure --prefix=$MONETDB_HOME --enable-embedded --enable-embedded-java

Then run make to build libs and make install to put them in place
$ make -j && make install

## Building embedded Java
Navigate to the main sources dir and then to java/embedded. Or most likely where this readmy file is. Run Maven build.
$ mvn clean install

The process uses the preset MONETDB_HOME variable for the location of the libraries. It will also run a set of unit test to validate the build.

## Alternativly
Run the build-all.sh found in the directory along with this file, supplying the location of the installation directory as the only argument.
$ build-all.sh <location to installation directory>

# Usage
After building it all, you can use the Java binaries. In the test dir you can see examples of how to use either the native columnar interface or the JDBC one.
Remember to set the MONETDB_HOME environmental variable and the -Djava.library.path flag, providing the MonetDB installation directory location. E.g.
-Djava.library.path=$MONETDB_HOME
