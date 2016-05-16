PUBSUB_STREAMS_SCHEMA = {
    "title": "JSON schema publish/subscribe streams",
    "description": "Validate data inserted",
    "$schema": "http://json-schema.org/draft-04/schema#",
    "type": "object",

    "anyOf": [{
        "properties": {
            "schema": {"type": "string"},
            "stream": {"type": "string"},
            "action": {"type": "string", "enum": ["sub", "subscribe", "desub", "desubscribe", "read"]},
        },
        "required": ["schema", "stream", "action"],
        "additionalProperties": False
    }, {
        "properties": {
            "action": {"type": "string", "enum": ["info"]}
        },
        "required": ["action"],
        "additionalProperties": False
    }]
}
