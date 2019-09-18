#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

#include "mlogger.h"

pthread_mutex_t mlog_mutex = PTHREAD_MUTEX_INITIALIZER;

char mlog_buffer[MLOG_BUFFER_SIZE];
int mlog_buffer_allocated_size = 0;

FILE *mlog_output;
char *mlog_file_name = MLOG_FILE_NAME;
int mlog_file_size = 0;
int mlog_file_id = 1;

LOG_LEVEL CUR_LOG_LEVEL = DEFAULT_LOG_LEVEL;
LOG_LEVEL CUR_FLUSH_LEVEL = DEFAULT_FLUSH_LEVEL;

int MLOG_STOP = FALSE;


// Internal error messages in mlogger
static void _mlog_error(const char *function_name, char *error_msg)
{
    fprintf(stderr, "Error in function %s: %s\n", function_name, error_msg);
    // exit(-1);
}


// Internal info messages in mlogger
static void _mlog_info(const char *function_name, char *error_msg)
{
    fprintf(stderr, "[%s] %s\n", function_name, error_msg);
}


// Check if log file is open
static void _mlog_file_is_open(void)
{
    if(!mlog_output)
    {
        _mlog_error(__func__, "The log file is not open");
        exit(-1);
    }
}


// Output error from snprintf of vsnprintf
static void _mlog_log_output_error(int bytes_written)
{
    if(bytes_written < 0)
    {
        _mlog_error(__func__, "Output error - Returned negative value");
        exit(-1);
    }
}


// Prepares a file in order to write the contents of the buffer 
// when necessary. The file name each time is merovingian_{int}.log
static void _mlog_create_file(void)
{
    char id[INT_MAX_LEN]; 
    snprintf(id, INT_MAX_LEN, "%d", mlog_file_id);

    char file_name[FILENAME_MAX];
    sprintf(file_name, "%s%s%s%s", mlog_file_name, "_", id, ".log");

    mlog_output = fopen(file_name, "a+");

    _mlog_file_is_open();
}


// Return the log level as string 
// static char* _mlog_level_as_string(LOG_LEVEL level)
// {
//     switch(level)
//     {
//         case M_NONE:
//             return "NONE";
//         case M_DEBUG:
//             return "DEBUG";
//         case M_INFO:
//             return "INFO";
//         case M_WARNING:
//             return "WARNING";
//         case M_ERROR:
//             return "ERROR";
//         case M_CRITICAL:
//             return "CRITICAL";
//     }
// }


static void _mlog_lock()
{
    pthread_mutex_lock(&mlog_mutex);
}


static void _mlog_unlock()
{
    pthread_mutex_unlock(&mlog_mutex);
}





/**
 * 
 * API CALLS
 * 
 */ 
void mlog_init(void)
{
    _mlog_info(__func__, "Starting mlogger");
    _mlog_create_file();
}


void mlog_stop(void)
{
    MLOG_STOP = TRUE;
    mlog_flush_buffer();
    fclose(mlog_output);
}


void mlog_set_log_level(LOG_LEVEL level)
{
    if(CUR_LOG_LEVEL != M_NONE && level == M_NONE)
    {
        mlog_flush_buffer();
    }

    _mlog_lock();
    { 
        CUR_LOG_LEVEL = level;
    }
    _mlog_unlock();
}


void mlog_reset_log_level(void)
{  
    if(CUR_LOG_LEVEL != M_NONE)
    {
        mlog_flush_buffer();
    }

    _mlog_lock();
    { 
        CUR_LOG_LEVEL = M_NONE;
    }
    _mlog_unlock();
}


void mlog_set_flush_level(LOG_LEVEL level)
{
    _mlog_lock();
    { 
        CUR_FLUSH_LEVEL = level;
    }
    _mlog_unlock();
}


void mlog_reset_flush_level(void)
{
    _mlog_lock();
    { 
        CUR_FLUSH_LEVEL = M_ERROR;
    }
    _mlog_unlock();
}


char* mlog_timestamp(void)
{
    static char datetime[20];
    time_t now = time(NULL);
    struct tm *tmp = localtime(&now);
    strftime(datetime, sizeof(datetime), "%Y-%m-%d %H:%M:%S", tmp);

    return datetime;
}


// TODO -> Rewrite this
void mlog_log(LOG_LEVEL level, int event_id, const char *fmt, ...)
{
    _mlog_file_is_open();
    
    if(level >= CUR_LOG_LEVEL && CUR_LOG_LEVEL > M_NONE)
    {
        _mlog_lock();
        
            va_list va;

            // Calculate the remaining buffer space
            int buffer_space = MLOG_BUFFER_SIZE - mlog_buffer_allocated_size;
            int retry_buffer_fill = TRUE;

            // snprintf(char *str, size_t count, ...) -> including null terminating character
            va_start(va, fmt);
            int bytes_written = vsnprintf(mlog_buffer + mlog_buffer_allocated_size, buffer_space, fmt, va);
            va_end(va);

            _mlog_log_output_error(bytes_written);

            // snprintf returned value -> does not include the null terminating character
            bytes_written++;

            // Message fits the buffer
            if(bytes_written < buffer_space)
            {
                // Increase the current buffer size
                mlog_buffer_allocated_size += bytes_written;
                retry_buffer_fill = FALSE;
            }
        
        _mlog_unlock();


        // Message did not fit in buffer
        if(retry_buffer_fill == TRUE)
        {
            mlog_flush_buffer();

            _mlog_lock();
            
                va_start(va, fmt);
                bytes_written = vsnprintf(mlog_buffer + mlog_buffer_allocated_size, buffer_space, fmt, va);
                va_end(va);
                
                _mlog_log_output_error(bytes_written);

                // Message is too big, to fit the empty buffer
                // Write it directly to the file
                if(bytes_written >= buffer_space)
                {
                    vfprintf(mlog_output, fmt, va);
                    fflush(mlog_output);
                }
                else 
                {
                    // Written to buffer
                    bytes_written++;
                    mlog_buffer_allocated_size += bytes_written;
                }
            
            _mlog_unlock();
        }

        // Flush the buffer in case the event is important depending on the flush-level
        if(event_id >= (int) CUR_FLUSH_LEVEL)
        {
            mlog_flush_buffer();
        }
    }
}


void mlog_flush_buffer(void)
{
    _mlog_lock();
    {
        fwrite(&mlog_buffer, mlog_buffer_allocated_size, 1, mlog_output);
        fflush(mlog_output);
        
        // Increase file size tracking
        mlog_file_size += mlog_buffer_allocated_size;

        // Reset buffer
        memset(mlog_buffer, 0, MLOG_BUFFER_SIZE);
        mlog_buffer_allocated_size = 0;
    }
    _mlog_unlock();

    // Even if the existing file is full, the logger should not create
    // a new file in case mlog_stop has been called
    if (mlog_file_size >= MLOG_MAX_FILE_SIZE && MLOG_STOP == FALSE)
    {
        fclose(mlog_output);
        mlog_file_size = 0;
        mlog_file_id++;
        _mlog_create_file();
    }
}
