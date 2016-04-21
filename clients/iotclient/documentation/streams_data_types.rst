.. _data_types:

********************
Available Data Types
********************

The following list contains the available data types for columns creation on the stream engine. The list was made as much compatible with `MonetDB's data types <https://www.monetdb.org/Documentation/Manuals/SQLreference/Datatypes>`_. For each column definiton, one of the types from the list must be provided along with its name.	

By default a column is not nullable, but the pair :code:`"nullable": true` can be provided to make it nullable. There is also the possibility to add a default value if the column value is not present on a JSON insert tuple (ex: :code:`"default": "text"`). The default value is validated against the column's validation during the stream creation.

.. warning:: A column cannot be nullable and have a default value simultaneously.

.. important:: The provided keys as well as the types values on the JSON object during the creation request must be lowercase.

Text Types
==========

Text, String, CLOB, Character Large Object
------------------------------------------

MonetDB's string types with unbounded length. The insertion must be provided as a JSON string.

Char, Character, Varchar, Character Varying
-------------------------------------------

MonetDB's string types with a bounded length. The :code:`limit` parameter must be provided as an interger. The insertion must be provided as a JSON string within the limit.

UUID
----

An *Universally Unique Identifier* according to `RFC 4122 <https://www.ietf.org/rfc/rfc4122.txt>`_. The insertion as a JSON string is validated against the regular expression :code:`^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`.

MAC
---

A *Media Access Control Address* identifier. The insertion as a JSON string is validated against the regular expression :code:`^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$`. As MonetDB doesn't have a MAC equivalent data type yet, the data is stored as :code:`char(17)`.

URL
---

An *Uniform Resource Locator* as a specific type of an URI is validated according to `RFC 3987 <https://www.ietf.org/rfc/rfc3987.txt>`_. The insertion must also be a JSON String.

Inet
----

An *IPv4* address. The insertion as a JSON string is validated against the regular expression :code:`^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$` with further semantic validation.

InetSix
-------

An *IPv6* address. The value must be a JSON String. As MonetDB doesn't have an Ipv6 equivalent data type yet, the data is stored as :code:`char(45)`.

.. warning:: Currently this validation is made with Python *socket.inet_pton()* function that is currently available only in some UNIX systems. `Check the documentation for further details <https://docs.python.org/2.7/library/socket.html#socket.inet_pton>`_.

Regex
-----

A string allways validated with a provided regular expression. The JSON must contain a :code:`regex` key with the regular expression. As MonetDB doesn't have a Regex equivalent data type yet, the data is stored as :code:`string`.

Enum
----

A SQL :code:`CHAR` type validated against a pre-defined array. During creation, the :code:`values` key must be present with a JSON array of Strings containing the values of the enum (ex: :code:`"values": ["red", "blue", "green"]`). The default value if provided must be one of the values.

Number Types
============

For all number types, a minimum and maximum values can be added for validation.

Tinyint, Smallint, Int, Integer, Bigint, Hugeint
------------------------------------------------

Signed integers. The type name specifies its bit capacity. If the value is grater than the bit capacity, it will be truncated. The insertion must be provided as a JSON integer.

+--------------+----------+
| **Type**     | **Size** |
+--------------+----------+
| Tinyint      |   8      |
+--------------+----------+
| Smallint     |  16      |
+--------------+----------+
| Int, Integer |  32      |
+--------------+----------+
| Bigint       |  64      |
+--------------+----------+
| Hugeint      | 128      |
+--------------+----------+

Real, Float, Double
-------------------

Floating point numbers. The type name specifies its bit capacity. If the value is grater than the bit capacity, it will be truncated. Real is 32 bit, while Float and Double are 64 bit. The insertion must be provided as a JSON float.

Decimal, Numeric
----------------

Numbers with a specific precision and scale. The precision must be between 1 and 38 (default 18), and the scale between 0 and the precision (default 0). The insertion must be provided as a JSON float.

Boolean Type
============

Boolean
-------

A :code:`true` or :code:`false` value. The inserted column has to be a JSON boolean.

Date and Time Types
===================

As in numbers, a minimum and maximum values can be added for validation. The inserted column must be a JSON String according to the type's regular expression.

Date
----

A regular date in the Gregorian Calendar, expressed in format :code:`YYYY-MM-DD`.

Time
----

The time of day expressed in format :code:`HH:MM:SS.sss`. If the pair :code:`"timezone": false` is provided, the timezone will be truncated.

Timestamp
---------

A timestamp according to `RFC 3339 <https://www.ietf.org/rfc/rfc3339.txt>`_ with timezone. The regular expression is the standard `ISO 8601 with timezone format <https://en.wikipedia.org/wiki/ISO_8601>`_. If the pair :code:`"timezone": false` is provided, the timezone will be truncated.
