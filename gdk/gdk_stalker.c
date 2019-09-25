#include <assert.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "monetdb_config.h"
#include "gdk.h"
#include "gdk_stalker.h"

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

gdk_stalker_buffer stalker = { .allocated_size = 0 };
bool GDK_STALKER_STOP = false;

FILE *output_file;
int file_size = 0;
int file_id = 1;

ATOMIC_TYPE CUR_LOG_LEVEL = DEFAULT_LOG_LEVEL;
ATOMIC_TYPE CUR_FLUSH_LEVEL = DEFAULT_FLUSH_LEVEL;

// Internal info messages in GDKstalker
static void _GDKstalker_info(const char *function_name, char *error_msg)
{
    fprintf(stderr, "[%s] %s\n", function_name, error_msg);
}

// Output error from snprintf of vsnprintf
static void _GDKstalker_log_output_error(int bytes_written)
{
    assert(bytes_written >= 0);
}


// Check if log file is open
static void _GDKstalker_file_is_open(void)
{
    assert(output_file);
}


// Prepares a file in order to write the contents of the buffer 
// when necessary. The file name each time is merovingian_{int}.log
static void _GDKstalker_create_file(void)
{
    char id[INT_MAX_LEN]; 
    snprintf(id, INT_MAX_LEN, "%d", file_id);

    char file_name[FILENAME_MAX];
    sprintf(file_name, "%s%s%s%s", FILE_NAME, "_", id, ".log");

    output_file = fopen(file_name, "a+");

    _GDKstalker_file_is_open();
}



/**
 * 
 * API CALLS
 * 
 */ 
char* GDKstalker_timestamp(void)
{
    static char datetime[20];
    time_t now = time(NULL);
    struct tm *tmp = localtime(&now);
    strftime(datetime, sizeof(datetime), "%Y-%m-%d %H:%M:%S", tmp);

    return datetime;
}


gdk_return GDKstalker_init(void)
{
    _GDKstalker_info(__func__, "Starting GDKstalker");
    _GDKstalker_create_file();
    return GDK_SUCCEED;
}


gdk_return GDKstalker_stop(void)
{
    GDK_STALKER_STOP = true;
    return GDKstalker_flush_buffer();
}


gdk_return GDKstalker_set_log_level(LOG_LEVEL level)
{
    if(CUR_LOG_LEVEL == level)
        return GDK_SUCCEED;

    if(level == M_NONE && CUR_LOG_LEVEL != M_NONE)
    {
        int GDK_result = GDKstalker_flush_buffer();
        if(GDK_result == GDK_FAIL)
            return GDK_FAIL;
    }

    ATOMIC_SET(&CUR_LOG_LEVEL, level);

    return GDK_SUCCEED;
}


gdk_return GDKstalker_reset_log_level(void)
{  
    if(CUR_LOG_LEVEL == M_NONE)
        return GDK_SUCCEED;
   
    int GDK_result = GDKstalker_flush_buffer();
    if(GDK_result == GDK_FAIL)
        return GDK_FAIL;

    ATOMIC_SET(&CUR_LOG_LEVEL, M_NONE);

    return GDK_SUCCEED;
}


gdk_return GDKstalker_set_flush_level(LOG_LEVEL level)
{
    if(CUR_FLUSH_LEVEL == level)
        return GDK_SUCCEED;

    ATOMIC_SET(&CUR_FLUSH_LEVEL, level);

    return GDK_SUCCEED;
}


gdk_return GDKstalker_reset_flush_level(void)
{
    if(CUR_FLUSH_LEVEL == M_ERROR)
        return GDK_SUCCEED;

    ATOMIC_SET(&CUR_FLUSH_LEVEL, M_ERROR);

    return GDK_SUCCEED;
}


// TODO -> Rewrite this
gdk_return GDKstalker_log(LOG_LEVEL level, int event_id, const char *fmt, ...)
{
    _GDKstalker_file_is_open();
    
    if(level >= CUR_LOG_LEVEL && CUR_LOG_LEVEL > M_NONE)
    {
        pthread_mutex_lock(&mutex);
        
            va_list va;

            // Calculate the remaining buffer space
            int buffer_space = BUFFER_SIZE - stalker.allocated_size;
            bool retry_buffer_fill = true;

            // snprintf(char *str, size_t count, ...) -> including null terminating character
            va_start(va, fmt);
            int bytes_written = vsnprintf(stalker.buffer + stalker.allocated_size, buffer_space, fmt, va);
            va_end(va);

            _GDKstalker_log_output_error(bytes_written);

            // snprintf returned value -> does not include the null terminating character
            bytes_written++;

            // Message fits the buffer
            if(bytes_written < buffer_space)
            {
                // Increase the current buffer size
                stalker.allocated_size += bytes_written;
                retry_buffer_fill = false;
            }
        
        pthread_mutex_unlock(&mutex);


        // Message did not fit in buffer
        if(retry_buffer_fill)
        {
            int GDK_result = GDKstalker_flush_buffer();
            if(GDK_result == GDK_FAIL)
                return GDK_FAIL;

            pthread_mutex_lock(&mutex);
            
                va_start(va, fmt);
                bytes_written = vsnprintf(stalker.buffer + stalker.allocated_size, buffer_space, fmt, va);
                va_end(va);
                
                _GDKstalker_log_output_error(bytes_written);

                // Message is too big, to fit the empty buffer
                if(bytes_written >= buffer_space)
                {
                    // Do nothing in this case
                }
                else 
                {
                    // Written to buffer
                    bytes_written++;
                    stalker.allocated_size += bytes_written;
                }
            
            pthread_mutex_unlock(&mutex);
        }

        // Flush the buffer in case the event is important depending on the flush-level
        if(event_id >= (int) CUR_FLUSH_LEVEL)
        {
            int GDK_result = GDKstalker_flush_buffer();
            if(GDK_result == GDK_FAIL)
                return GDK_FAIL;
        }
    }

    return GDK_SUCCEED;
}


gdk_return GDKstalker_flush_buffer(void)
{
    pthread_mutex_lock(&mutex);
    {
        fwrite(&stalker.buffer, stalker.allocated_size, 1, output_file);
        fflush(output_file);
        
        // Increase file size tracking
        file_size += stalker.allocated_size;

        // Reset buffer
        memset(stalker.buffer, 0, BUFFER_SIZE);
        stalker.allocated_size = 0;
    }
    pthread_mutex_unlock(&mutex);

    // Even if the existing file is full, the logger should not create
    // a new file in case GDKstalker_stop has been called
    if (file_size >= MAX_FILE_SIZE && !GDK_STALKER_STOP)
    {
        fclose(output_file);
        file_size = 0;
        file_id++;
        _GDKstalker_create_file();
    }
    else if(GDK_STALKER_STOP)
    {
        fclose(output_file);
    }
    
    return GDK_SUCCEED;
}
