-- This Source Code Form is subject to the terms of the Mozilla Public
-- License, v. 2.0.  If a copy of the MPL was not distributed with this
-- file, You can obtain one at http://mozilla.org/MPL/2.0/.
--
-- Copyright 1997 - July 2008 CWI, August 2008 - 2016 MonetDB B.V.

-------------------------------------------------------------------------
----------------------- GRID related functions --------------------------
-------------------------------------------------------------------------
CREATE FILTER FUNCTION distance(x1 TINYINT,  y1 TINYINT,  x2 TINYINT,  y2 TINYINT,  distance double) EXTERNAL NAME grid.distance;
CREATE FILTER FUNCTION distance(x1 SMALLINT, y1 SMALLINT, x2 SMALLINT, y2 SMALLINT, distance double) EXTERNAL NAME grid.distance;
CREATE FILTER FUNCTION distance(x1 INTEGER,  y1 INTEGER,  x2 INTEGER,  y2 INTEGER,  distance double) EXTERNAL NAME grid.distance;
CREATE FILTER FUNCTION distance(x1 BIGINT,   y1 BIGINT,   x2 BIGINT,   y2 BIGINT,   distance double) EXTERNAL NAME grid.distance;
CREATE FILTER FUNCTION distance(x1 DECIMAL,  y1 DECIMAL,  x2 DECIMAL,  y2 DECIMAL,  distance double) EXTERNAL NAME grid.distance;
CREATE FILTER FUNCTION distance(x1 REAL,     y1 REAL,     x2 REAL,     y2 REAL,     distance double) EXTERNAL NAME grid.distance;
CREATE FILTER FUNCTION distance(x1 DOUBLE,   y1 DOUBLE,   x2 DOUBLE,   y2 DOUBLE,   distance double) EXTERNAL NAME grid.distance;
