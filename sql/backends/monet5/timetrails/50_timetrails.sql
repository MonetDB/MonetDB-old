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

create schema timetrails;

-- register and start a continuous query
create procedure timetrails.query(qry string, maxcalls integer)
	external name timetrails.query;

create procedure timetrails.query(qry string)
	external name timetrails.query;

create procedure timetrails.query("schema" string, name string, maxcalls integer)
	external name timetrails.query;

create procedure timetrails.query("schema" string, name string)
	external name timetrails.query;

create procedure timetrails.resume("schema" string, name string)
	external name timetrails.resume;

create procedure timetrails.resume()
	external name timetrails.resume;

create procedure timetrails.pause("schema" string, name string)
	external name timetrails.pause;

create procedure timetrails.keep("schema" string, name string)
	external name timetrails.keep;

create procedure timetrails.release("schema" string, name string)
	external name timetrails.release;

create procedure timetrails.pause()
	external name timetrails.pause;

create procedure timetrails.wait(ms integer)
	external name timetrails.wait;

create procedure timetrails.stop()
	external name timetrails.stop;

create procedure timetrails.deregister("schema" string, name string)
	external name timetrails.deregister;

create procedure timetrails.deregister()
	external name timetrails.deregister;

-- set the scheduler periodic delay
create procedure timetrails.period(n integer)
	external name timetrails.period;

-- deliver a new basket with tuples

create procedure timetrails.heartbeat("schema" string, "table" string, msec integer)
	external name timetrails.heartbeat;

create procedure timetrails.heartbeat("schema" string, "table" string, msec bigint)
	external name timetrails.heartbeat;

-- cleanup activities
create procedure timetrails.tumble("schema" string, "table" string, elem integer)
	external name timetrails.tumble;

create procedure timetrails.window("schema" string, "table" string, elem integer)
	external name timetrails.window;

create procedure timetrails.cycles("schema" string, "query" string, elem integer)
	external name timetrails.cycles;

-- Inspection tables
create function timetrails.gettumble("schema" string, "table" string) returns integer
external name timetrails.gettumble;

create function timetrails.getwindow("schema" string, "table" string) returns integer
external name timetrails.getwindow;

create function timetrails.getheartbeat("schema" string, "table" string) returns bigint
external name timetrails.getheartbeat;

create function timetrails.baskets()
returns table( "schema" string, "table" string, "status" string, winsize int, winstride int, timeslice int, timestride int, heartbeat int, seen timestamp, "count" bigint, events bigint)
external name timetrails.baskets;

create procedure timetrails.show("schema" string, "query" string)
external name timetrails.show;

create function timetrails.queries()
 returns table( "schema" string, "function" string, "status" string, lastrun timestamp, runs int, avgtime bigint, error string)
 external name timetrails.queries;

create function timetrails.inputs()
 returns table( "s" string, "t" string, "sch" string, "qry" string)
 external name timetrails.inputplaces;

create function timetrails.outputs()
 returns table( "s" string, "t" string, "sch" string, "qry" string)
 external name timetrails.outputplaces;

create function timetrails.errors()
returns table( "table" string, error string)
external name timetrails.errors;

-- tables for timetrailswebserver
CREATE TABLE timetrails.webserverstreams (table_id INTEGER, base TINYINT, "interval" INTEGER NULL, unit CHAR(1) NULL);
