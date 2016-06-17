.. _introduction:

************
Introduction
************

This application is a Python web server destined to extend MonetDB RDBMS to the Internet-of-Things while using a streaming engine. This project is derived from Pedro Ferreira master thesis, a junior researcher at CWI. The objective of the master thesis to build a streaming extension of MonetDB to the IOT world, which is obtaining much demand nowadays. To accomplish this, an IOT topology around MonetDB is being built, where this server aims to analyse the input of streams. At the same time, another web server is being built to analyze the streams' output.

This web server is built using Python programming language, version 2.7, using a `Simple Websockets server <https://github.com/dpallot/simple-websocket-server>`_ . The required packages on requirements.txt should be installed before running the server.

The server creates a polling connection with MonetDB, listening for every new output queries. The Streaming context will then be constantly updated during the polling connection. A small publisher/subscriber pattern was added to the server where the clients can to subscribe to notified right away when an output basket is created. It is also possible to perform small queries on the created data for pagination while including an offset or a limit of the number of tuples to retrieve. All the communication process is assured using a full-duplex WebSockets connection.

The following chapters explain the server parameters and the Websockets protocol used.
