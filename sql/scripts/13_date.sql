-- This Source Code Form is subject to the terms of the Mozilla Public
-- License, v. 2.0.  If a copy of the MPL was not distributed with this
-- file, You can obtain one at http://mozilla.org/MPL/2.0/.
--
-- Copyright 1997 - July 2008 CWI, August 2008 - 2018 MonetDB B.V.

create system function str_to_date(s string, format string) returns date
	external name mtime."str_to_date";

create system function date_to_str(d date, format string) returns string
	external name mtime."date_to_str";

create system function str_to_time(s string, format string) returns time
	external name mtime."str_to_time";

create system function time_to_str(d time, format string) returns string
	external name mtime."time_to_str";

create system function str_to_timestamp(s string, format string) returns timestamp
	external name mtime."str_to_timestamp";

create system function timestamp_to_str(d timestamp, format string) returns string
	external name mtime."timestamp_to_str";

grant execute on function str_to_date to public;
grant execute on function date_to_str to public;
grant execute on function str_to_time to public;
grant execute on function time_to_str to public;
grant execute on function str_to_timestamp to public;
grant execute on function timestamp_to_str to public;
