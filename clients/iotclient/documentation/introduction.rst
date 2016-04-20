.. _instroduction:


************
Introduction
************

This appplication is a Python web server destinated to extend MonetDB RDBMS to the Internet-of-Things while using a streaming engine. This project is derived from Pedro Ferreira master thesis, a junior researcher at CWI. The objective of the master thesis to re-evaluate the DataCell extension of MonetDB to the IOT world, which was postponed for several years since its release. To accomplish this, an IOT topology around MonetDB is being built, where this server aims to analyse the input of streams. Later it will also be included another web server to analyze the output of streams.

This web server is built using Python programming language, version 2.7, using `Flask-RESTful <https://pypi.python.org/pypi/Flask-RESTful>`_ framework. The required packages on requirements.txt should be installed before running the server. 

The web server is capable of creating and deleting streams for the renewed DataCell extension. After a stream is created, it's possible to make batch inserts on it. The requests are made using RESTfull requests with JSON content. Both stream creation and stream insertion are validated using `JSON Schema Draft 4 <http://json-schema.org/documentation.html>`_. On a batch insert if a tuple is invalid, then no tuples are inserted. The stream engine adds an implicit timestamp column to recognize when the tuple was inserted. The inserted tuples are later evaluated using MonetDB's relational engine with continuous queries.

The following chapters explain the server parameters, RESTful resources and how to manage the streams.
