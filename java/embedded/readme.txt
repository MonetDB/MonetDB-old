# Building
The embedded Java version of MonetDB requires a few options to be set in order to build the required libraries and NOT build some of the unneeded binaries.

## Setting environmental variables
In order for the embedded Java to find the MonetDB libraries we are currently using an environmental variables. The same one would be used as a prefix when configuring the MonetDB build (see blow in 'Building MonetDB).
$ export MONETDB_HOME=<location to installation directory>

## Building MonetDB
Navigate the the main sources dir and bootstrap the build.
$ sh bootstrap
Create a BUILD directory and run the configure script found in the main sources dir. Set a prefix location using
$ ../configure --prefix=$MONETDB_HOME --enable-embedded --enable-embedded-java

Then run make to build libs and make install to put them in place
$ make -j && make install

## Building embedded Java
Navigate to the main sources dir and then to java/embedded. Or most likely where this readmy file is. Run Maven build.
$ mvn clean install

The process uses the preset MONETDB_HOME variable for the location of the libraries. It will also run a set of unit test to validate the build.

## Alternatively
Run the build-all.sh found in the directory along with this file, supplying the location of the installation directory as the only argument.
$ build-all.sh <location to installation directory>

# Usage
After building it all, you can use the Java binaries. In the test dir you can see examples of how to use either the native columnar interface or the JDBC one.
Remember to set the MONETDB_HOME environmental variable and the -Djava.library.path flag, providing the MonetDB installation directory location. E.g.
-Djava.library.path=$MONETDB_HOME

# Implementation
The embedded Java version of MonetDB is heavily based and dependent on the generic one (or MonetDBLite). The embedded Java uses the C-level interface in `tools/embedded`, particularly the `embedded` library. In a few words, that interface exposes a few functions for starting, querying and other action. So far we only use: `monetdb_startup` and `monetdb_query`, together with `monetdb_connect`, when required.

## JNI C code
To interface Java with C we use JNI. JNI code comes with two side - Java and native (C) code. In the Java code we declare a function `native`, which indicates that it is actually implemented in C. We then write the native implementation. This is where we call the embedded C-leve interface function from the generic embedded version.

### org_monetdb_embedded_MonetDBEmbedded
- Java_org_monetdb_embedded_MonetDBEmbedded_startupWrapper
This is the wrapper function for starting up a database. It requires a location of the libs, a working directory for the database itself. It call the native interface function, and starts up the database. There are also a few checks in place, making sure that only one instance of the process can be started within the same Java/JVM process.

- Java_org_monetdb_embedded_MonetDBEmbedded_queryWrapper
This is the query function. All it needs is a string with an SQL query. It passes it to the the native implementation and gets back a pointer to the result set. From there some meta data about the result set is extracted and passed back to Java in a new object (org.monetdb.embedded.result.EmbeddedQueryResult). The metadata is: column names, column types, number of columns and number of rows. In addition the object stores a long, that is the pointer to the result set struct. This is used to later extract data from the native results set and convert it to Java structures (see below).

### org_monetdb_embedded_result_EmbeddedQueryResult
- Java_org_monetdb_embedded_result_EmbeddedQueryResult_getColumnWrapper
The is the function that will transfer actual data from C to Java. It gets the long pointer stored in the org.monetdb.embedded.result.EmbeddedQueryResult object, which points to the result set, and the id of the column to retrieve. Based on the column type a different static function is called, creating and returning a Java object, that matches the column type. The supported column types are: boolean, byte, short, integer, long, float, double and string.

- Getting columns
For each column we create two structures, one native Java array (e.g for int - an int array) and a boolean array for the null values. We the check if the column has nulls. If it does, we iterate over all values and copy them to the Java array, check if the values isn't. For each null, we write true in the boolean array. The two arrays are then passed to the constructor for the encapsulating Java column object.

- Java_org_monetdb_embedded_result_EmbeddedQueryResult_cleanupResult
This function is called only when the result set object is closed in Java. We used the stored result set pointer and call the native result clean-up fuction.

## Java columnar interface
### org.monetdb.embedded.MonetDBEmbedded
- This is the object that represents a MonetDB database. It statically loads the `embedded_java` library, which contains the above native JNI code. We've also put in place a few safeguards, reducing the chance of starting a database twice for the same object, if one's running.

### org.monetdb.embedded.result.EmbeddedQueryResult
This is the meta object that a native query creates and returns. It holds only metadata and calling getColumn() will interface with C and actually get the column.

### org.monetdb.embedded.result.column
Each column extends the abstract class `org.monetdb.embedded.result.column.Column`. When the native code returns a columns, it create one of the object in the `org.monetdb.embedded.result.column` package. Each of them holds a (if possible) primitive-type array with the values and the Column abstract class holds a boolean array with the null flags. When getValue() is called, we check if the value at that row is null or otherwise we return a non-primitive matching type. E.g. for integer we return an Integer, s.t. null can be returned as nulls.

## JDBC interface
On top of the columnar interface, we've build a simple JDBC interface. This was done by extending the existing MonetDB JDBC driver and overriding a few as possible functions. The overrides facilitate moving from using the MCL library (communicatiing tuples over MAPI) to using the embedded columnar interface described above. The JDBC can be found in the `nl.cwi.monetdb.jdbc` packge in the project.

There are a few ways to use the interface:
- Using JDBC connection object.
In that case, once can create a `MonetDBEmbeddedConnection`, passing as properties only the directory of the database:
```
Properties props = new Properties();
props.put("database", "/tmp/dbtest");
Connection connection = new MonetDBEmbeddedConnection(props)```
This process will create a `org.monetdb.embedded.MonetDBEmbedded` object and start the database.

Then create statement and get the result set, as one would in JDBC:
```
Statement statement = connection.createStatement()
statement.execute("SELECT * FROM test;");
ResultSet result = statement.getResultSet();```
This will query the embedded database and get the meta data of the result set.
Calling get value, will then get the desired column and row in order from the `org.monetdb.embedded.result.column.Column` object.

- Using the embedded/columnar interfacing
Create a database object:
```
final Path directoryPath = Files.createTempDirectory("monetdbtest");
datbaseDirectory = directoryPath.toFile();
MonetDBEmbedded db = new MonetDBEmbedded(datbaseDirectory);
db.start()```

Executer a query and get the JDBC result set from the columnar result set (meta) object:
```
EmbeddedQueryResult result = db.query("SELECT 1;");
MonetDBEmbeddedResultSet jdbcResultSet = result.getJDBCResultSet();```
From here one, the result set can be treated as (almost) any other JDBC result set. It is technically the same object as the one in the above example.
