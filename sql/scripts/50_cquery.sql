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

-- This is the first interface for continuous queries

create schema cquery;

create procedure cquery.new(sch string, cqname string, qrytxt string)
	external name cquery;
create procedure cquery.new(cqname string, qrytxt string)
	external name cquery;

create procedure cquery.resume()
	external name cquery.resume;
create procedure cquery.resume(cqname string)
	external name cquery.resume;

create procedure cquery.pause()
	external name cquery.pause;
create procedure cquery.pause(cqname string)
	external name cquery.pause;

create procedure cquery.release()
	external name cquery.release;

-- The following commands can be part of the cquery itself
create procedure cquery.wait(ms integer)
	external name cquery.wait;

-- set the scheduler periodic delay
create procedure cquery.period(n integer)
	external name cquery.period;

create function cquery.summary()
 returns table( "schema" string, "function" string, "status" string, lastrun timestamp, runs int, avgtime bigint, error string)
 external name cquery.summary;
create function cquery.log()
 returns table(tick timestamp,  "schema" string, "function" string, "status" string, time bigint, errors string)
 external name cquery.summary;


create function cquery.show(qryname string)
returns string
external name cquery.show;

-- The stream table support functions.

create procedure cquery.heartbeat("schema" string, "table" string, msec integer)
	external name cquery.heartbeat;

create procedure cquery.heartbeat("schema" string, "table" string, msec bigint)
	external name cquery.heartbeat;

-- cleanup activities
create procedure cquery.tumble("schema" string, "table" string, elem integer)
	external name cquery.tumble;

create procedure cquery.window("schema" string, "table" string, elem integer)
	external name cquery.window;

create procedure cquery.cycles("schema" string, "query" string, elem integer)
	external name cquery.cycles;

-- Inspection tables
create function cquery.gettumble("schema" string, "table" string) returns integer
external name cquery.gettumble;

create function cquery.getwindow("schema" string, "table" string) returns integer
external name cquery.getwindow;

create function cquery.getheartbeat("schema" string, "table" string) returns bigint
external name cquery.getheartbeat;

create function streams.baskets()
returns table( "schema" string, "table" string, "status" string, winsize int, winstride int, timeslice int, timestride int, heartbeat int, seen timestamp, "count" bigint, events bigint)
external name streams.baskets;

create function streams.inputs()
 returns table( "s" string, "t" string, "sch" string, "qry" string)
 external name streams.inputplaces;

create function streams.outputs()
 returns table( "s" string, "t" string, "sch" string, "qry" string)
 external name streams.outputplaces;
