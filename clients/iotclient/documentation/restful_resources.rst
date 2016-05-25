.. _restful_resources:

*****************
RESTful Resources
*****************

Bellow is listed the available RESTful resources for both servers on the server. In case of error, the server will return the error message with the respective HTTP response code.

.. important:: All the keywords provided on JSON must be lower case, as well the column data types.

Administration Server
=====================

The administration server provides resources to create and delete streams. Should be listening on the host only.

.. _streams:

/streams
--------

**GET**

Returns a JSON file with details about all the streams currently created on the webserver. For each stream besides its schema and name, it provides the currently number of tuples inserted on the baskets per column, description of columns (`See data types <streams_data_types.html#data_types>`__), the flushing method (`See streams creation for details <streams_creation.html#creating_streams>`__). An example is shown bellow:

.. code-block:: json

	[
	  {
	    "tuples_inserted_per_basket": 1,
	    "columns": [
	      {
		"type": "real",
		"name": "temperature",
		"nullable": false
	      },
	      {
		"type": "text",
		"name": "sensorid",
		"nullable": false
	      }
	    ],
	    "flushing": {
	      "base": "tuple",
	      "number": 50
	    },
	    "stream": "temperature",
	    "schema": "measures"
	  }
	]

/context
--------

**POST**

Creates a stream using a pre-defined JSON schema. The JSON must include the stream's schema, the stream's name, the stream's columns and the flushing method. The flushing can be either time based, tuple based or automatic (:code:`auto`). For tuple based flushing, the number of tuples to flush must be provided using the :code:`number` field. In time based flushing, the :code:`interval` field tells the time units between flushes and the :code:`unit` field must be "s", "m" or "h" for seconds, minutes or hours respectively. In automatic flushing, the baskets are flushed whenver a new batch is inserted. For columns `see data types for details <streams_data_types.html#data_types>`__.

Bellow is the JSON used to create the stream in streams_:

.. code-block:: json

	{
	  "schema": "measures",
	  "stream": "temperature",
	  "flushing": {
	    "base": "tuple",
	    "number": 50
	    },
	  "columns": [
	    {
	      "type": "real",
	      "name": "temperature",
	      "nullable": false
	    },
	    {
	      "type": "text",
	      "name": "sensorid",
	      "nullable": false
	    }
	  ]
	}

**DELETE**

Deletes an existing stream. Only the stream's schema and name are required. To delete the stream in streams_ provide the following JSON:

.. code-block:: json

	{
	  "schema": "measures",
	  "stream": "temperature"
	}

Application Server
==================

The application server provides resources to make insertions on streams. Should be listening to all interfaces.

/streams
--------

**GET**

Same resource as streams_ .

/stream/<schema_name>/<stream_name>
-----------------------------------

**POST**

Insert a batch of tuples on the provided stream in the URL. The insert must be an array of JSON object with pairs of column-value. All tuples are validated according to the defined data types on each column. If there is an invalid tuple, none of the tuples is inserted. The implicit timestamp and the host identifier are automatically added. Bellow is shown an example to insert several tuples on the stream in streams_:

.. code-block:: json

	[
	  {
	    "sensorid": "living room",
	    "temperature": 32.6
	  },
	  {
	    "sensorid": "kitchen",
	    "temperature": 34.2
	  },
	  {
	    "sensorid": "bathroom",
	    "temperature": 28.9
	  }
	]
