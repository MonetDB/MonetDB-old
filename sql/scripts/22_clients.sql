-- This Source Code Form is subject to the terms of the Mozilla Public
-- License, v. 2.0.  If a copy of the MPL was not distributed with this
-- file, You can obtain one at http://mozilla.org/MPL/2.0/.
--
-- Copyright 1997 - July 2008 CWI, August 2008 - 2018 MonetDB B.V.

create system function sys.password_hash (username string)
	returns string
	external name sql.password;

create system function sys.sessions()
returns table("user" string, "login" timestamp, "sessiontimeout" bigint, "lastcommand" timestamp, "querytimeout" bigint, "active" bool)
external name sql.sessions;
create view sys.sessions as select * from sys.sessions();

create system procedure sys.shutdown(delay tinyint)
external name sql.shutdown;

create system procedure sys.shutdown(delay tinyint, force bool)
external name sql.shutdown;

-- control the query and session time out
create system procedure sys.settimeout("query" bigint)
	external name clients.settimeout;
create system procedure sys.settimeout("query" bigint, "session" bigint)
	external name clients.settimeout;
create system procedure sys.setsession("timeout" bigint)
	external name clients.setsession;
