-- This Source Code Form is subject to the terms of the Mozilla Public
-- License, v. 2.0.  If a copy of the MPL was not distributed with this
-- file, You can obtain one at http://mozilla.org/MPL/2.0/.
--
-- Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.

CREATE SCHEMA logging;

-- Flush GDKstalker buffer
CREATE PROCEDURE logging.flush()
       EXTERNAL NAME logging.flush;

-- Set GDKstalker log level
CREATE PROCEDURE logging.setloglevel(lvl INT)
       EXTERNAL NAME logging.setloglevel;

-- Reset the log level of GDKstalker 
CREATE PROCEDURE logging.resetloglevel()
       EXTERNAL NAME logging.resetloglevel;

-- Set GDKstalker flush level
CREATE PROCEDURE logging.setflushlevel(lvl INT)
       EXTERNAL NAME logging.setflushlevel;

-- Reset the flush level of GDKstalker 
CREATE PROCEDURE logging.resetflushlevel()
        EXTERNAL NAME logging.resetflushlevel;
