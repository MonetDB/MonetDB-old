/* 
  - Since you probably want to lock the memory buffer while writting to file, it makes sense to create a second one to continue 
    collecting the log records while the first one is written. Once a buffer is full, you switch their role.
  - Clean messages & replace with API Calls (except from fprintf(stderr) - maybe also Mfprintf?)
  - Logging lever per component (ALGODEBUG etc.)
  - Control logger through SQL
*/

#define INT_MAX_LEN ((__CHAR_BIT__ * sizeof(int) - 1) / 3  + 2)
#define MLOG_BUFFER_SIZE 64000                                          // 64KB

#define DEFAULT_LOG_LEVEL 0                                             // M_NONE
#define DEFAULT_FLUSH_LEVEL 200                                         // M_ERROR

#define MLOG_FILE_NAME "merovingian"
#define MLOG_MAX_FILE_SIZE 1073741824                                   // 1 GB

#define TRUE 1
#define FALSE 0

/**
 *
 *      NONE      0 
 *      DEBUG     1 - 99        Following the DEBUG settings in the source code
 *      INFO      100 - 149     Reserved for package specific debugging
 *      WARNING   150 - 199
 *      ERROR     200 - 254     Package specific errors (GDK, MAL, SQL, MAPI)
 *      CRITICAL  255           Cannot be ignored
 * 
 */
 
// The minimum code is assigned to each LOG_LEVEL
typedef enum { M_NONE = 0, M_DEBUG = 1, M_INFO = 100, M_WARNING = 150, M_ERROR = 200, M_CRITICAL = 255 } LOG_LEVEL;



/**
 *  MLogger API
 */

// Initialize logger - basically creates the logging file
void mlog_init(void);


// Sets the log level to one of the enum LOG_LEVELS above. If the current log level 
// is not NONE and mlog_set_log_level sets it to NONE we flush the buffer first 
// in order to "discard" the messages that are there from the previous log levels
void mlog_set_log_level(LOG_LEVEL level);


// Resets the log level to the default one - NONE. If the current log level is not NONE 
// and mlog_reset_log_level() is called we need to flush the buffer first in order to 
// "discard" messages kept from other log levels
void mlog_reset_log_level(void);


// Sets the minimum flush level that an event will trigger the logger to flush the buffer
void mlog_set_flush_level(LOG_LEVEL level);


// Resets the flush level to the default (ERROR)
void mlog_reset_flush_level(void);


// Flush the buffer to the file. If after flushing the buffer, the file is greater 
// or equal to 1 GB, then we close the current one and we swap to a new one increasing the ID
void mlog_flush_buffer(void);


// Flushes the contents of the buffer and closes the log file
void mlog_stop(void);


// Returns the timestamp in the form of datetime
char* mlog_timestamp(void);


// TODO -> Write comments
void mlog_log(LOG_LEVEL level, int event_id, const char *fmt, ...);
