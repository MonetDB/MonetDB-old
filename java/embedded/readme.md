# Building
The embedded Java version of MonetDB requires a few options to be set in order to build the required libraries and NOT build some of the unneeded binaries.

Run the `build-all.sh` found in the directory along with this file. This script will build the embedded version on MonetDB (and only it), pull all libraries in a single fat lib, copy to the specified dir, build and test the embedded Java

## Manually
### Building MonetDB
Navigate the the main sources dir and bootstrap the build.
$ sh bootstrap
Create a BUILD directory and run the configure script found in the main sources dir. Set a prefix location using
```
../configure --prefix=<someplace> --enable-embedded --enable-embedded-java```

Then run make to build libs and make install to put them in place
```
make -j```

After that compile all the libraries into a single library for embedded use
```
OFILES=`find common clients/mapilib/ gdk monetdb5/mal monetdb5/modules monetdb5/optimizer sql tools/embedded java/embedded -name "*.o" | tr "\n" " "`
gcc -shared -o libmonetdb5.dylib $OFILES -lpthread -lpcre -lbz2 -llzma -lcurl -lz -liconv```

### Building embedded Java
Navigate to the main sources dir and then to java/embedded. Or most likely where this readme file is. Run Maven build.
```
mvn clean install```

# Usage
After building it all, you can use the Java binaries. In the test dir you can see examples of how to use either the native columnar interface or the JDBC one.

## Libraries
Packed in the `src/main/resources/lib` dir of the `.jar` there should be a library that contains the native (C) part of MonetDB. The extension of the library should either be the default for a dynamic libraries on your OS or the generic (for JNI) `.jnilib`. The code is designed to make use of the library shipped with the `.jar`, if another `libmonetdb5` is not provided.

In an IDE or when Maven runs tests from command line, the application will use the unpacked library, already present in the `src/main/resources/lib` dir (since there isn't a `.jar` yet). When running "in production" - from a `.jar`, the application will stream copy the library to a temp dir, and load the library from there. This is needed, since one cannot use the packed libraries in a `.jar` directly.

### Alternatively
If you want to use another library for the native embedded MonetDB implementation, you have to set `-Djava.library.path=<location>` flag, providing location of the directory the all-in-one embedded MonetDB lib is.

## Java columnar interface
### org.monetdb.embedded.MonetDBEmbedded
- This is the object that represents a MonetDB database. It statically loads the `embedded_java` library, which contains the above native JNI code. We've also put in place a few safeguards, reducing the chance of starting a database twice for the same object, if one's running.

### org.monetdb.embedded.result.EmbeddedQueryResult
This is the meta object that a native query creates and returns. It holds only metadata and calling `getColumn()` will interface with C and actually get the column.

### org.monetdb.embedded.result.column
Each column extends the abstract class `org.monetdb.embedded.result.column.Column`. When the native code returns a columns, it create one of the object in the `org.monetdb.embedded.result.column` package. Each of them holds a (if possible) primitive-type array with the values and the Column abstract class holds a boolean array with the null flags. When `getValue()` is called, we check if the value at that row is null or otherwise we return a non-primitive matching type. E.g. for integer we return an Integer, s.t. null can be returned as nulls.

## JDBC interface
On top of the columnar interface, we've build a simple JDBC interface. This was done by extending the existing MonetDB JDBC driver and overriding a few as possible functions. The overrides facilitate moving from using the MCL library (communicatiing tuples over MAPI) to using the embedded columnar interface described above. The JDBC can be found in the `nl.cwi.monetdb.jdbc` package in the project.

There are a few ways to use the interface:
### Using JDBC connection object.
- In that case, once can create a `MonetDBEmbeddedConnection`, passing as properties only the directory of the database:
```
Properties props = new Properties();
props.put("database", "/tmp/dbtest");
Connection connection = new MonetDBEmbeddedConnection(props)```
This process will create a `org.monetdb.embedded.MonetDBEmbedded` object and start the database.

- Then create statement and get the result set, as one would in JDBC:
```
Statement statement = connection.createStatement()
statement.execute("SELECT * FROM test;");
ResultSet result = statement.getResultSet();```
This will query the embedded database and get the meta data of the result set.
- Calling get value, will then get the desired column and row in order from the `org.monetdb.embedded.result.column.Column` object.

### Using the embedded/columnar interfacing
- Create a database object:
```
final Path directoryPath = Files.createTempDirectory("monetdbtest");
datbaseDirectory = directoryPath.toFile();
MonetDBEmbedded db = new MonetDBEmbedded(datbaseDirectory);
db.start()```

- Execute a query and get the JDBC result set from the columnar result set (meta) object:
```
EmbeddedQueryResult result = db.query("SELECT 1;");
MonetDBEmbeddedResultSet jdbcResultSet = result.getJDBCResultSet();```

- From here one, the result set can be treated as (almost) any other JDBC result set. It is technically the same object as the one in the above example.

# Implementation
The embedded Java version of MonetDB is heavily based and dependent on the generic one (or MonetDBLite). The embedded Java uses the C-level interface in `tools/embedded`. In a few words, that interface exposes a few functions for starting, querying and other action. So far we only use: `monetdb_startup()` and `monetdb_query()`, together with `monetdb_connect()`, when required.

## JNI C code
To interface Java with C we use JNI. JNI code comes with two complementing parts - Java and native (C in our case) code. In the Java code we declare a function `native`, which indicates that it is actually implemented in C. We then write the native implementation. This is where we call the embedded C-leve interface function from the generic embedded version.

### org_monetdb_embedded_MonetDBEmbedded
- `Java_org_monetdb_embedded_MonetDBEmbedded_startupWrapper`
This is the wrapper function for starting up a database. It requires a location of the libs, a working directory for the database itself. It call the native interface function, and starts up the database. There are also a few checks in place, making sure that only one instance of the process can be started within the same Java/JVM process.

- `Java_org_monetdb_embedded_MonetDBEmbedded_queryWrapper`
This is the query function. All it needs is a string with an SQL query. It passes it to the the native implementation and gets back a pointer to the result set. From there some meta data about the result set is extracted and passed back to Java in a new object (`org.monetdb.embedded.result.EmbeddedQueryResult`). The metadata is: column names, column types, number of columns and number of rows. In addition the object stores a long, that is the pointer to the result set struct. This is used to later extract data from the native results set and convert it to Java structures (see below).

### org_monetdb_embedded_result_EmbeddedQueryResult
- `Java_org_monetdb_embedded_result_EmbeddedQueryResult_getColumnWrapper`
The is the function that will transfer actual data from C to Java. It gets the long pointer stored in the `org.monetdb.embedded.result.EmbeddedQueryResult` object, which points to the result set, and the id of the column to retrieve. Based on the column type a different static function is called, creating and returning a Java object, that matches the column type. The supported column types are: boolean, byte, short, integer, long, float, double and string.

- Getting columns
For each column we create two structures, one native Java array (e.g for int - an int array) and a boolean array for the null values. We the check if the column has nulls. If it does, we iterate over all values and copy them to the Java array, check if the values isn't. For each null, we write true in the boolean array. The two arrays are then passed to the constructor for the encapsulating Java column object.

- `Java_org_monetdb_embedded_result_EmbeddedQueryResult_cleanupResult`
This function is called only when the result set object is closed in Java. We used the stored result set pointer and call the native result clean-up function.

## JDBC interface
The classes for the JDBC driver are in the `nl.cwi.monetdb.jdbc` space, since a lot of the methods in the original driver were implemented as package private. A few changes to the original JDBC were also needed, mostly setting different visibility for methods and a few constructor changes. As a result, the JDBC part of the embedded MonetDB only works with the JDBC code that is currently part of the same brach on the source tree.
