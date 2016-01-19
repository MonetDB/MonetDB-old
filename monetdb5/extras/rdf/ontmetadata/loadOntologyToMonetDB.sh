#dbpedia3.9
NUMMETADATA=`cat ontMetadata.dbpedia39.csv | wc -l`
NUMATTRIBUTES=`cat ontAttribute.dbpedia39.csv | wc -l`

cp loadOntologySAMPLE.sql loadtmp.sql
sed -i "s:NUMMETADATA:$NUMMETADATA:g" loadtmp.sql
sed -i "s:NUMATTRIBUTES:$NUMATTRIBUTES:g" loadtmp.sql
sed -i "s:MetaFile:${PWD}/ontMetadata.dbpedia39.csv:g" loadtmp.sql
sed -i "s:AttFile:${PWD}/ontAttribute.dbpedia39.csv:g" loadtmp.sql

mclient < loadtmp.sql

#goodrelations
NUMMETADATA=`cat ontMetadata.gr.csv | wc -l`
NUMATTRIBUTES=`cat ontAttribute.gr.csv | wc -l`

cp loadOntologySAMPLE.sql loadtmp.sql
sed -i "s:NUMMETADATA:$NUMMETADATA:g" loadtmp.sql
sed -i "s:NUMATTRIBUTES:$NUMATTRIBUTES:g" loadtmp.sql
sed -i "s:MetaFile:${PWD}/ontMetadata.gr.csv:g" loadtmp.sql
sed -i "s:AttFile:${PWD}/ontAttribute.gr.csv:g" loadtmp.sql


mclient < loadtmp.sql

#lubm
NUMMETADATA=`cat ontMetadata.lubm.csv | wc -l`
NUMATTRIBUTES=`cat ontAttribute.lubm.csv | wc -l`

cp loadOntologySAMPLE.sql loadtmp.sql
sed -i "s:NUMMETADATA:$NUMMETADATA:g" loadtmp.sql
sed -i "s:NUMATTRIBUTES:$NUMATTRIBUTES:g" loadtmp.sql
sed -i "s:MetaFile:${PWD}/ontMetadata.lubm.csv:g" loadtmp.sql
sed -i "s:AttFile:${PWD}/ontAttribute.lubm.csv:g" loadtmp.sql


mclient < loadtmp.sql


#eurostat
NUMMETADATA=`cat ontMetadata.eurostat.csv | wc -l`
NUMATTRIBUTES=`cat ontAttribute.eurostat.csv | wc -l`

cp loadOntologySAMPLE.sql loadtmp.sql
sed -i "s:NUMMETADATA:$NUMMETADATA:g" loadtmp.sql
sed -i "s:NUMATTRIBUTES:$NUMATTRIBUTES:g" loadtmp.sql
sed -i "s:MetaFile:${PWD}/ontMetadata.eurostat.csv:g" loadtmp.sql
sed -i "s:AttFile:${PWD}/ontAttribute.eurostat.csv:g" loadtmp.sql


mclient < loadtmp.sql


#swrc
NUMMETADATA=`cat ontMetadata.swrc.csv | wc -l`
NUMATTRIBUTES=`cat ontAttribute.swrc.csv | wc -l`

cp loadOntologySAMPLE.sql loadtmp.sql
sed -i "s:NUMMETADATA:$NUMMETADATA:g" loadtmp.sql
sed -i "s:NUMATTRIBUTES:$NUMATTRIBUTES:g" loadtmp.sql
sed -i "s:MetaFile:${PWD}/ontMetadata.swrc.csv:g" loadtmp.sql
sed -i "s:AttFile:${PWD}/ontAttribute.swrc.csv:g" loadtmp.sql


mclient < loadtmp.sql

#foaf
NUMMETADATA=`cat ontMetadata.foaf.csv | wc -l`
NUMATTRIBUTES=`cat ontAttribute.foaf.csv | wc -l`

cp loadOntologySAMPLE.sql loadtmp.sql
sed -i "s:NUMMETADATA:$NUMMETADATA:g" loadtmp.sql
sed -i "s:NUMATTRIBUTES:$NUMATTRIBUTES:g" loadtmp.sql
sed -i "s:MetaFile:${PWD}/ontMetadata.foaf.csv:g" loadtmp.sql
sed -i "s:AttFile:${PWD}/ontAttribute.foaf.csv:g" loadtmp.sql


mclient < loadtmp.sql

#bsbm
NUMMETADATA=`cat ontMetadata.bsbm.csv | wc -l`
NUMATTRIBUTES=`cat ontAttribute.bsbm.csv | wc -l`

cp loadOntologySAMPLE.sql loadtmp.sql
sed -i "s:NUMMETADATA:$NUMMETADATA:g" loadtmp.sql
sed -i "s:NUMATTRIBUTES:$NUMATTRIBUTES:g" loadtmp.sql
sed -i "s:MetaFile:${PWD}/ontMetadata.bsbm.csv:g" loadtmp.sql
sed -i "s:AttFile:${PWD}/ontAttribute.bsbm.csv:g" loadtmp.sql


mclient < loadtmp.sql

#rdfvocabulary
NUMMETADATA=`cat ontMetadata.rdfvocabulary.csv | wc -l`
NUMATTRIBUTES=`cat ontAttribute.rdfvocabulary.csv | wc -l`

cp loadOntologySAMPLE.sql loadtmp.sql
sed -i "s:NUMMETADATA:$NUMMETADATA:g" loadtmp.sql
sed -i "s:NUMATTRIBUTES:$NUMATTRIBUTES:g" loadtmp.sql
sed -i "s:MetaFile:${PWD}/ontMetadata.rdfvocabulary.csv:g" loadtmp.sql
sed -i "s:AttFile:${PWD}/ontAttribute.rdfvocabulary.csv:g" loadtmp.sql


mclient < loadtmp.sql

#Open Graph Protocol (ogp)
NUMMETADATA=`cat ontMetadata.ogp.csv | wc -l`
NUMATTRIBUTES=`cat ontAttribute.ogp.csv | wc -l`

cp loadOntologySAMPLE.sql loadtmp.sql
sed -i "s:NUMMETADATA:$NUMMETADATA:g" loadtmp.sql
sed -i "s:NUMATTRIBUTES:$NUMATTRIBUTES:g" loadtmp.sql
sed -i "s:MetaFile:${PWD}/ontMetadata.ogp.csv:g" loadtmp.sql
sed -i "s:AttFile:${PWD}/ontAttribute.ogp.csv:g" loadtmp.sql


mclient < loadtmp.sql

#opengraphschema
NUMMETADATA=`cat ontMetadata.opengraphschema.csv | wc -l`
NUMATTRIBUTES=`cat ontAttribute.opengraphschema.csv | wc -l`

cp loadOntologySAMPLE.sql loadtmp.sql
sed -i "s:NUMMETADATA:$NUMMETADATA:g" loadtmp.sql
sed -i "s:NUMATTRIBUTES:$NUMATTRIBUTES:g" loadtmp.sql
sed -i "s:MetaFile:${PWD}/ontMetadata.opengraphschema.csv:g" loadtmp.sql
sed -i "s:AttFile:${PWD}/ontAttribute.opengraphschema.csv:g" loadtmp.sql


mclient < loadtmp.sql

#Dublin core
NUMMETADATA=`cat ontMetadata.dc.csv | wc -l`
NUMATTRIBUTES=`cat ontAttribute.dc.csv | wc -l`

cp loadOntologySAMPLE.sql loadtmp.sql
sed -i "s:NUMMETADATA:$NUMMETADATA:g" loadtmp.sql
sed -i "s:NUMATTRIBUTES:$NUMATTRIBUTES:g" loadtmp.sql
sed -i "s:MetaFile:${PWD}/ontMetadata.dc.csv:g" loadtmp.sql
sed -i "s:AttFile:${PWD}/ontAttribute.dc.csv:g" loadtmp.sql


mclient < loadtmp.sql

#schema.org
NUMMETADATA=`cat ontMetadata.schema.org.csv | wc -l`
NUMATTRIBUTES=`cat ontAttribute.schema.org.csv | wc -l`

cp loadOntologySAMPLE.sql loadtmp.sql
sed -i "s:NUMMETADATA:$NUMMETADATA:g" loadtmp.sql
sed -i "s:NUMATTRIBUTES:$NUMATTRIBUTES:g" loadtmp.sql
sed -i "s:MetaFile:${PWD}/ontMetadata.schema.org.csv:g" loadtmp.sql
sed -i "s:AttFile:${PWD}/ontAttribute.schema.org.csv:g" loadtmp.sql


mclient < loadtmp.sql

#dbpediaUmbel: Link between Dbpedia and Umbel
NUMMETADATA=`cat ontMetadata.dbpediaUmbel.csv | wc -l`
NUMATTRIBUTES=`cat ontAttribute.dbpediaUmbel.csv | wc -l`

head -n 1 loadOntologySAMPLE.sql > loadtmp.sql
sed -i "s:NUMMETADATA:$NUMMETADATA:g" loadtmp.sql
sed -i "s:MetaFile:${PWD}/ontMetadata.dbpediaUmbel.csv:g" loadtmp.sql

if [ "$NUMATTRIBUTES" -gt 0 ]
then
	tail -n 1 loadOntologySAMPLE.sql >> loadtmp.sql
	sed -i "s:NUMATTRIBUTES:$NUMATTRIBUTES:g" loadtmp.sql
	sed -i "s:AttFile:${PWD}/ontAttribute.dbpediaUmbel.csv:g" loadtmp.sql
else 	
	echo "Zero attributes"
fi

mclient < loadtmp.sql

#List of possible ontologies
NUMONT=`cat ontList.csv | wc -l`

cp loadOntologyListSAMPLE.sql loadtmp.sql
sed -i "s:NUMONT:$NUMONT:g" loadtmp.sql
sed -i "s:OntListFile:${PWD}/ontList.csv:g" loadtmp.sql


mclient < loadtmp.sql
