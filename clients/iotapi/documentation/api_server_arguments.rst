.. _starting_webapi:

********************
Starting the web api
********************

The packages listened on :code:`requirements.txt` file must be installed before running the server. To avoid conflicts with other versions of the packages existing on the host machine, is recommended to create a Python virtual environment for the server. `About Python virtual environments <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_.

The server starts as a regular Python program invoking the :code:`main.py` file, however there are several arguments that can be passed to change its behavior.

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

Set the filesystem directory where the baskets will be created. By default in UNIX systems is on :code:`/etc/iotapi` directory, while on Windows is on the directory where the :code:`main.py` script was invoked.

**-l  - -log=**

Location of logfile. On the logfile is reported when streams are created or removed, when tuples are inserted and when the baskets are flushed. By default in UNIX systems is :code:`/var/log/iot/iotapi.log`, while on Windows is the :code:`iotapi.log` on the directory where the :code:`main.py` script was called.


Web API Listening and Behavior
------------------------------

Customize the Web API server parameters and behavior.

**-sh  - -shost=**

Listening host of the application (IOT) server. By default is on :code:`0.0.0.0`.

**-sp  - -sport=**

Listening port of the application (IOT) server. By default is on port :code:`8002`.

**-pi  - -polling=**

Set the polling interval to MonetDB database for updates. By default is :code:`60` seconds.

Database Connection
-------------------

Credentials for the MAPI connection to MonetDB database.

.. note:: The user's password will be prompted during the initialization of the server.

**-d  - -host=**

Listening host of MonetDB database. By default is on :code:`127.0.0.1`.

**-d  - -port=**

Listening port of MonetDB database. By default is on port :code:`50000`.

**-u  - -user=**

Name of the user to authenticate. By default is user :code:`monetdb`.

**-d  - -database=**

Name of database to use. By default is :code:`iotdb` database.
