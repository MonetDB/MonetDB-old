NUMMETADATA=`cat ontMetadata.dbpedia.csv | wc -l`
NUMATTRIBUTES=`cat ontAttribute.dbpedia.csv | wc -l`

cp loadOntologySAMPLE.sql loadtmp.sql
sed -i "s:NUMMETADATA:$NUMMETADATA:g" loadtmp.sql
sed -i "s:NUMATTRIBUTES:$NUMATTRIBUTES:g" loadtmp.sql
sed -i "s:MetaFile:${PWD}/ontMetadata.dbpedia.csv:g" loadtmp.sql
sed -i "s:AttFile:${PWD}/ontAttribute.dbpedia.csv:g" loadtmp.sql




mclient < loadtmp.sql



NUMMETADATA=`cat ontMetadata.gr.csv | wc -l`
NUMATTRIBUTES=`cat ontAttribute.gr.csv | wc -l`

cp loadOntologySAMPLE.sql loadtmp.sql
sed -i "s:NUMMETADATA:$NUMMETADATA:g" loadtmp.sql
sed -i "s:NUMATTRIBUTES:$NUMATTRIBUTES:g" loadtmp.sql
sed -i "s:MetaFile:${PWD}/ontMetadata.gr.csv:g" loadtmp.sql
sed -i "s:AttFile:${PWD}/ontAttribute.gr.csv:g" loadtmp.sql


mclient < loadtmp.sql


NUMONT=`cat ontList.csv | wc -l`

cp loadOntologyListSAMPLE.sql loadtmp.sql
sed -i "s:NUMONT:$NUMONT:g" loadtmp.sql
sed -i "s:OntListFile:${PWD}/ontList.csv:g" loadtmp.sql


mclient < loadtmp.sql
