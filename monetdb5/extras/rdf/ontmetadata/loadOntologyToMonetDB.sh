NUMMETADATA=`cat ontMetadata.dbpedia.csv | wc -l`
NUMATTRIBUTES=`cat ontAttribute.dbpedia.csv | wc -l`

cp loadOntologySAMPLE.sql loadtmp.sql
sed -i "s:NUMMETADATA:$NUMMETADATA:g" loadtmp.sql
sed -i "s:NUMATTRIBUTES:$NUMATTRIBUTES:g" loadtmp.sql

mclient -d dbpedia --port=50000 < loadtmp.sql



NUMMETADATA=`cat ontMetadata.gr.csv | wc -l`
NUMATTRIBUTES=`cat ontAttribute.gr.csv | wc -l`

cp loadOntologySAMPLE.sql loadtmp.sql
sed -i "s:NUMMETADATA:$NUMMETADATA:g" loadtmp.sql
sed -i "s:NUMATTRIBUTES:$NUMATTRIBUTES:g" loadtmp.sql

mclient -d dbpedia --port=50000 < loadtmp.sql
