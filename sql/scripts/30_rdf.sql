-- The contents of this file are subject to the MonetDB Public License
-- Version 1.1 (the "License"); you may not use this file except in
-- compliance with the License. You may obtain a copy of the License at
-- http://www.monetdb.org/Legal/MonetDBLicense
--
-- Software distributed under the License is distributed on an "AS IS"
-- basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
-- License for the specific language governing rights and limitations
-- under the License.
--
-- The Original Code is the MonetDB Database System.
--
-- The Initial Developer of the Original Code is CWI.
-- Portions created by CWI are Copyright (C) 1997-July 2008 CWI.
-- Copyright August 2008-2015 MonetDB B.V.
-- All Rights Reserved.

-- SQL statements to make the RDF Relational Storage Schema query-able from
-- the sql frontend.

-- It only needs to be run once after a fresh installation of MonetDB/SQL.

-- create an RDF schema
create schema rdf;

-- create a graph_name|id table
create table rdf.graph (gname string, gid int);

-- create a procudure to load an RDF document
-- the schema string should be removed in the future and auto-fill it from
-- the backend
create procedure rdf_shred(location string, gname string, sch string)
	external name sql.rdfshred;

create procedure rdf_retrieveSubschema(oldschema string, count int, sch string)
	external name sql.rdfretrievesubschema;

create procedure rdf_schema_explore(tbname string, clname string)
	external name rdf.rdfschemaexplore;

create procedure rdf_loadOntology(location string, sch string)
	external name rdf.rdfloadontology;

create procedure rdf_reorganize(sch string, tbname string, threshold int, expmode int)
	external name sql.rdfreorganize; 

create procedure rdf_scan(query string, sch string)	
	external name sql.rdfscan;

create procedure rdf_deserialize()
	external name sql.rdfdeserialize; 

create procedure rdf_prepare()
	external name sql.rdfprepare; 

--create procedure rdf_idtostr(id oid)
--	external name sql.rdfidtostr;
	
--create procedure rdf_strtoid(urlstr string)
--	external name sql.rdfstrtoid;

create function rdf_idtostr(id oid)
	returns string external name sql.rdfidtostr;
	
create function rdf_strtoid(urlstr string)
	returns oid external name sql.rdfstrtoid;

create function rdf_timetoid(datetime string)
	returns oid external name sql.rdftimetoid;

create function tkzr_strtoid(urlstr string) 
	returns oid external name tokenizer.locate;	

create function tkzr_idtostr(id oid) 
	returns string external name tokenizer.take;	
