-- This Source Code Form is subject to the terms of the Mozilla Public
-- License, v. 2.0.  If a copy of the MPL was not distributed with this
-- file, You can obtain one at http://mozilla.org/MPL/2.0/.
--
-- Copyright 2008-2015 MonetDB B.V.
--
-- Acknowledgement
-- ===============
-- 
-- The research leading to this code has been partially funded by the European
-- Commission under FP7 programme project #611068.


CREATE PROCEDURE precommit(id bigint)
	EXTERNAL name sql.precommit;

CREATE PROCEDURE persistcommit(id bigint)
	EXTERNAL name sql.persistcommit;

CREATE PROCEDURE forcepersistcommit(id bigint)
	EXTERNAL name sql.forcepersistcommit;

CREATE PROCEDURE htmgarbagecollect(ts int)
	EXTERNAL name sql.htmgarbagecollect;

CREATE FUNCTION gethtmgarbagecollect() RETURNS integer
	EXTERNAL name sql.gethtmgarbagecollect;
