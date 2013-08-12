COPY NUMMETADATA RECORDS INTO ontmetadata  FROM '/export/scratch2/linnea/scripts/loadOntology/ontMetadata.csv'         USING DELIMITERS '|', '\n';
COPY NUMATTRIBUTES RECORDS INTO ontattributes FROM '/export/scratch2/linnea/scripts/loadOntology/ontAttribute.csv'     USING DELIMITERS '|', '\n';
