.. _starting_webserver:


***********************
Starting the web server
***********************

The server is started as a regular Python program invoking the :code:`main.py` file, however ther are several arguments that can be passed to change its behaviour.

::

    python main.py [arguments]

.. _arguments:

Arguments
=========

The possible arguments for the web server are listed bellow.

Paths
-----

.. important:: Make sure the current user has permissions on the locations provided.

**-f - -filesystem=**

Set the filesystem directory where the baskets will be created. By default in UNIX systems is on :code:`/etc/iotcollector` directory, while on Windows is on the directory where the :code:`main.py` script was invoked. While running the web server creates a :code:`baskets` directory.

**-l  - -logfile=**

Location of logfile. On the logfile is reported when streams are created or removed, when tuples are inserted and when the baskets are flushed. By default in UNIX systems is :code:`/var/log/iot/iot.log`, while on Windows is the :code:`iot.log` on the directory where the :code:`main.py` script was invoked.

**-c  - -configfile=**

Path of JSON file where the information of existing streams on the server will be stored. By default is the :code:`config.json` file on filesystem directory.


Host Identifier
---------------

If the *useidentifier* parameter is provided, an extra column on streams will be added with a custom name of the host for later identification. 

**-u  - -useidentifier=**

Boolean to indicate if the extra column will be added.

**-n  - -name=**

Custom name of the host. By default is the host's MAC address.


Web Server Listening
--------------------

For security purposes, two web servers are created by the application. On the administration server are present operations to create and delete streams, while on application server the insert operations are present instead. The administration server should be listening on the host only, while the application server should be listening to the outside.

**-ih  - -ihost=**

Listening host of the application (IOT) server. By default is on :code:`0.0.0.0`.

**-ip  - -iport=**

Listening port of the application (IOT) server. By default is on port :code:`8000`.

**-ah  - -ahost=**

Listening host of the administration (IOT) server. By default is on :code:`127.0.0.1`.

**-ap  - -aport=**

Listening port of the administration (IOT) server. By default is on port :code:`8001`.


Database Connection
-------------------

Credentials for the MAPI connection to MonetDB database.

.. note:: The user password will be prompted during the initialization of the server.

**-dh  - -dhostname=**

Listening host of MonetDB database. By default is on :code:`127.0.0.1`.

**-dp  - -dport=**

Listening port of MonetDB database. By default is on port :code:`50000`.

**-du  - -duser=**

Name of the user to authenticate. By default is user :code:`monetdb`.

**-dd  - -ddatabase=**

Name of database to use. By default is :code:`iotdb` database.


