-- This Source Code Form is subject to the terms of the Mozilla Public
-- License, v. 2.0.  If a copy of the MPL was not distributed with this
-- file, You can obtain one at http://mozilla.org/MPL/2.0/.
--
-- Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.

CREATE SCHEMA logging;

-- Flush GDKtracer buffer
CREATE PROCEDURE logging.flush()
        EXTERNAL NAME logging.flush;

-- Set GDKtracer log level for a specific component
CREATE PROCEDURE logging.setcomploglevel(comp INT, lvl INT)
        EXTERNAL NAME logging.setcomploglevel;

-- Reset the log level of GDKtracer for a specific component
CREATE PROCEDURE logging.resetcomploglevel(comp INT)
        EXTERNAL NAME logging.resetcomploglevel;

-- Set GDKtracer flush level
CREATE PROCEDURE logging.setflushlevel(lvl INT)
       EXTERNAL NAME logging.setflushlevel;

-- Reset the flush level of GDKtracer
CREATE PROCEDURE logging.resetflushlevel()
       EXTERNAL NAME logging.resetflushlevel;

-- Set the adapter of GDKtracer
CREATE PROCEDURE logging.setadapter(adapter INT)
       EXTERNAL NAME logging.setadapter;

-- Reset the adapter of GDKtracer
CREATE PROCEDURE logging.resetadapter()
       EXTERNAL NAME logging.resetadapter;
