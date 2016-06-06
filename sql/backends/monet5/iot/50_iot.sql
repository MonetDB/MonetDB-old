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
-- Copyright August 2008-2016 MonetDB B.V.
-- All Rights Reserved.

create schema iot;

-- register and start a continuous query
create procedure iot.query(qry string)
	external name iot.query;

create procedure iot.query("schema" string, name string)
	external name iot.query;

create procedure iot.activate("schema" string, name string)
	external name iot.activate;

create procedure iot.activate()
	external name iot.activate;

create procedure iot.deactivate("schema" string, name string)
	external name iot.deactivate;

create procedure iot.deactivate()
	external name iot.deactivate;

create procedure iot.deregister("schema" string, name string)
	external name iot.deregister;

create procedure iot.deregister()
	external name iot.deregister;

-- resume with limited the number of scheduler before next pause
create procedure iot.cycles(n integer)
	external name iot.cycles;

-- set the scheduler periodic delay
create procedure iot.period(n integer)
	external name iot.period;

-- deliver a new basket with tuples
create procedure iot.basket("schema" string, "table" string, dirpath string)
	external name iot.basket;

-- triggering conditions for each stream
create procedure iot.threshold("schema" string, "table" string, elem int)
	external name iot.threshold;

create procedure iot.beat("schema" string, "table" string, msec int)
	external name iot.beat;

-- cleaup activities 
create procedure iot.window("schema" string, "table" string, elem int)
	external name iot.window;

create procedure iot.window("schema" string, "table" string, elem int, slide int)
	external name iot.window;

-- Inspection tables
create function iot.baskets()
returns table( "schema" string, "table" string, "status" string, threshold int, winsize int, winstride int, timeslice int, timestride int, beat int, seen timestamp, events int)
external name iot.baskets;

create function iot.queries()
 returns table( "schema" string, "function" string, "status" string, lastrun timestamp, cycles int, events int, time bigint, error string)
 external name iot.queries;

create function iot.inputplaces()
 returns table( "s" string, "t" string, "sch" string, "qry" string)
 external name iot.inputplaces;

create function iot.outputplaces()
 returns table( "s" string, "t" string, "sch" string, "qry" string)
 external name iot.outputplaces;

-- create function iot.errors()
-- returns table( "schema" string,  "table" string, error string)
-- external name iot.errors;

-- tables for iotwebserver

CREATE TABLE iot.webserverflushing (table_id INTEGER, base TINYINT, "interval" INTEGER NULL, unit TINYINT NULL);

CREATE TABLE iot.webservervalidation (column_id INTEGER, special TINYINT NULL, validation1 string NULL, validation2 string NULL);

