-- This Source Code Form is subject to the terms of the Mozilla Public
-- License, v. 2.0.  If a copy of the MPL was not distributed with this
-- file, You can obtain one at http://mozilla.org/MPL/2.0/.
--
-- Copyright 1997 - July 2008 CWI, August 2008 - 2016 MonetDB B.V.

-- control the delay for oltp settings
create procedure sys.oltp_enable()
external name oltp.enable;

create procedure sys.oltp_disable()
external name oltp.disable;

create procedure sys.oltp_reset()
external name oltp.reset;

create function sys.oltp_locks()
returns table(
	started timestamp,
	username  string,
	lockid  integer,
	cnt	integer
)
external name oltp."table";

create function sys.oltp_is_enabled()
returns integer
external name oltp.isenabled;
