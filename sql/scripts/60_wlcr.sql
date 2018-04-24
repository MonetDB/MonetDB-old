-- This Source Code Form is subject to the terms of the Mozilla Public
-- License, v. 2.0.  If a copy of the MPL was not distributed with this
-- file, You can obtain one at http://mozilla.org/MPL/2.0/.
--
-- Copyright 1997 - July 2008 CWI, August 2008 - 2018 MonetDB B.V.

-- Workload Capture and Replay

-- Master commands
create system procedure master()
external name wlc.master;

create system procedure master(path string)
external name wlc.master;

create system procedure stopmaster()
external name wlc.stopmaster;

create system procedure masterbeat( duration int)
external name wlc."setmasterbeat";

create system function masterClock() returns string
external name wlc."getmasterclock";

create system function masterTick() returns bigint
external name wlc."getmastertick";

-- Replica commands
create system procedure replicate()
external name wlr.replicate;

create system procedure replicate(pointintime timestamp)
external name wlr.replicate;

create system procedure replicate(dbname string)
external name wlr.replicate;

create system procedure replicate(dbname string, pointintime timestamp)
external name wlr.replicate;

create system procedure replicate(dbname string, id tinyint)
external name wlr.replicate;

create system procedure replicate(dbname string, id smallint)
external name wlr.replicate;

create system procedure replicate(dbname string, id integer)
external name wlr.replicate;

create system procedure replicate(dbname string, id bigint)
external name wlr.replicate;

create system procedure replicabeat(duration integer)
external name wlr."setreplicabeat";

create system function replicaClock() returns string
external name wlr."getreplicaclock";

create system function replicaTick() returns bigint
external name wlr."getreplicatick";

