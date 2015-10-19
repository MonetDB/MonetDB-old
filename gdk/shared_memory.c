
#include "shared_memory.h"

#ifndef _WIN32

#include "gdk.h"
#include "gdk_private.h"
#include "../monetdb5/mal/mal_exception.h"
#include "mutils.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sched.h>
#include <errno.h>
#include <sys/sem.h>

static lng *shm_memory_ids;
static void **shm_ptrs;
static int shm_unique_id = 1;
static int shm_current_id = 0;
static int shm_max_id = 32;
static int shm_is_initialized = false;
static char shm_keystring[] = BINDIR;
static MT_Lock release_memory_lock;
static key_t base_key = 800000000;


str init_shared_memory(int id, size_t size, void **ptr, int flags);
void store_shared_memory(lng memory_id, void *ptr);
str release_shared_memory_id(int memory_id, void *ptr);

str init_mmap_memory(int id, size_t size, void **ptr, int flags);
str release_mmap_memory(void *ptr, size_t size);

int init_process_semaphore(int id, int count, int flags);

str initialize_shared_memory(void)
{
	if (shm_is_initialized) //maybe this should just return MAL_SUCCEED as well
        return createException(MAL, "shared_memory.init", "Attempting to initialize shared memory when it was already initialized.");

    //initialize the pointer to memory ID structure
	shm_ptrs = malloc(shm_max_id * sizeof(void*));
	shm_memory_ids = malloc(shm_max_id * sizeof(lng));
	shm_current_id = 0;
	shm_max_id = 32;
	shm_unique_id = 2;

    MT_lock_init(&release_memory_lock, "release_memory_lock");

    shm_is_initialized = true;
    return MAL_SUCCEED;
}

void store_shared_memory(lng memory_id, void *ptr)
{
    int i;

    assert(shm_is_initialized);


    for(i = 0; i < shm_current_id; i++)
    {
        if (shm_ptrs[i] == NULL)
        {
            shm_memory_ids[i] = memory_id;
            shm_ptrs[i] = ptr;
            return;
        }
    }

	if (shm_current_id >= shm_max_id)
	{
		void **new_ptrs = malloc(shm_max_id * 2 * sizeof(void*));
		lng *new_memory_ids = malloc(shm_max_id * 2 * sizeof(lng));

		memcpy(new_ptrs, shm_ptrs, sizeof(void*) * shm_max_id);
		memcpy(new_memory_ids, shm_memory_ids, sizeof(lng) * shm_max_id);

		free(shm_ptrs); free(shm_memory_ids);
		shm_ptrs = new_ptrs; shm_memory_ids = new_memory_ids;
		shm_max_id *= 2;
	}

	shm_memory_ids[shm_current_id] = memory_id;
	shm_ptrs[shm_current_id] = ptr;
	shm_current_id++;
}

int get_unique_shared_memory_id(int offset)
{
    int id;
    assert(shm_is_initialized);

	id = shm_unique_id;
	shm_unique_id += offset;
	return id;
}

str init_mmap_memory(int id, size_t size, void **return_ptr, int flags)
{   
    char address[100];
    void *ptr;
    int fd, result;
    snprintf(address, 100, "/tmp/temp_pyapi_mmap_%d", id);

    fd = open(address, flags | O_RDWR, MONETDB_MODE);
    if (fd < 0) {
        char *err = strerror(errno);
        errno = 0;
        close(fd);
        return createException(MAL, "shared_memory.get", "Could not create mmap file %s: %s", address, err);
    }
    if (flags != 0) {
        result = lseek(fd, size - 1, SEEK_SET);
        if (result == -1) {
            char *err = strerror(errno);
            errno = 0;
            close(fd);
            return createException(MAL, "shared_memory.get", "Failed to extend mmap file: %s", err);
        }
        result = write(fd, "", 1);
        if (result != 1) {
            char *err = strerror(errno);
            errno = 0;
            close(fd);
            return createException(MAL, "shared_memory.get", "Failed to write to mmap file: %s", err);
        }
    }
    ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (ptr == (void*) -1) {
        char *err = strerror(errno);
        errno = 0;
        return createException(MAL, "shared_memory.get", "Failure in mmap(NULL, %zu, PROT_WRITE, MAP_SHARED, %d, 0): %s", size, fd, err);
    }
    store_shared_memory(size, ptr);
    if (return_ptr != NULL) *return_ptr = ptr;
    return MAL_SUCCEED;
}

str release_mmap_memory(void *ptr, size_t size)
{
    int ret;
    ret = munmap(ptr, size);
    if (ret != 0) {
        char *err = strerror(errno);
        errno = 0;
        return createException(MAL, "shared_memory.release_mmap_memory", "Failure in munmap(%p,%zu): %s", ptr, size, err);
    }
    return MAL_SUCCEED;
}

str create_shared_memory(int id, size_t size, void **return_ptr)
{
    char *shared, *mmap;
    if ((shared = init_shared_memory(id, size, return_ptr, IPC_CREAT)) == MAL_SUCCEED) return MAL_SUCCEED;
    if ((mmap = init_mmap_memory(id, size, return_ptr, O_CREAT)) == MAL_SUCCEED) return MAL_SUCCEED;
    return createException(MAL, "shared_memory.release_mmap_memory", "Failed to create shared memory or mmap space.\nshared memory error: %s\nmmap error: %s", shared, mmap);
}

str get_shared_memory(int id, size_t size, void **return_ptr)
{
    char *shared, *mmap;
    if ((shared = init_shared_memory(id, size, return_ptr, 0)) == MAL_SUCCEED) return MAL_SUCCEED;
    if ((mmap = init_mmap_memory(id, size, return_ptr, 0)) == MAL_SUCCEED) return MAL_SUCCEED;
    return createException(MAL, "shared_memory.release_mmap_memory", "Failed to get shared memory or mmap space.\nshared memory error: %s\nmmap error: %s", shared, mmap);
}

str ftok_enhanced(int id, key_t *return_key);
str ftok_enhanced(int id, key_t *return_key)
{
    *return_key = base_key + id;
    return MAL_SUCCEED;
}

str init_shared_memory(int id, size_t size, void **return_ptr, int flags)
{
    lng shmid;
    void *ptr;
    int i;
    key_t key;
    str msg;

	msg = ftok_enhanced(id, &key);
    if (msg != MAL_SUCCEED)
    {
        return msg;
    }

	assert(shm_is_initialized);

	shmid = shmget(key, size, flags | 0666);
    if (shmid < 0)
    {
        char *err = strerror(errno);
        errno = 0;
        return createException(MAL, "shared_memory.get", "Error calling shmget(key:%zu,size:%zu,flags:%d): %s", (size_t)key, size, flags, err);
    }

    //check if the shared memory segment is already created, if it is we do not need to add it to the table and can simply return the pointer
    for(i = 0; i < shm_current_id; i++)
    {
        if (shm_memory_ids[i] == shmid)
        {
            if (return_ptr != NULL) *return_ptr = shm_ptrs[i];
            return MAL_SUCCEED;
        }
    }

	ptr = shmat(shmid, NULL, 0);
    if (ptr == (void*)-1)
    {
        char *err = strerror(errno);
        errno = 0;
        return createException(MAL, "shared_memory.get", "Error calling shmat(id:%lld,NULL,0): %s", shmid, err);
    }

    store_shared_memory(shmid, ptr);
    if (return_ptr != NULL) *return_ptr = ptr;
    return MAL_SUCCEED;
}

str release_shared_memory(void *ptr)
{
	int i = 0;
	int memory_id = -1;

    assert(shm_is_initialized);

    MT_lock_set(&release_memory_lock, "release_memory_lock");
	//find the memory_id accompanying the given pointer in the structure
	for(i = 0; i < shm_current_id; i++)
	{
		if (shm_ptrs[i] == ptr)
		{
            memory_id = shm_memory_ids[i];
            shm_memory_ids[i] = 0;
            shm_ptrs[i] = NULL;
			break;
		}
	}
    MT_lock_unset(&release_memory_lock, "release_memory_lock");

	assert(memory_id);
	//actually release the memory at the given ID
	if (release_shared_memory_id(memory_id, ptr) == MAL_SUCCEED) return MAL_SUCCEED;
    if (release_mmap_memory(ptr, memory_id) == MAL_SUCCEED) return MAL_SUCCEED;
    return createException(MAL, "shared_memory.release", "Failed to release shared memory.");
}

str release_shared_memory_id(int memory_id, void *ptr)
{
	if (shmctl(memory_id, IPC_RMID, NULL) == -1)
	{
        char *err = strerror(errno);
        errno = 0;
        return createException(MAL, "shared_memory.release", "Error calling shmctl(id:%d,IPC_RMID,NULL): %s", memory_id, err);
	}
	if (shmdt(ptr) == -1)
	{
        char *err = strerror(errno);
        errno = 0;
        return createException(MAL, "shared_memory.release", "Error calling shmdt(ptr:%p): %s", ptr, err);
	}
	return MAL_SUCCEED;
}

int init_process_semaphore(int id, int count, int flags)
{
    int key = ftok(shm_keystring, id);
    int semid = -1;
    if (key == (key_t) -1)
    {
        perror("ftok");
        return semid;
    }
    semid = semget(key, count, flags | 0666);
    if (semid < 0)
    {
        perror("semget failed: ");
    }
    return semid;
}

int create_process_semaphore(int id, int count)
{
    return init_process_semaphore(id, count, IPC_CREAT);
}

int get_process_semaphore(int sem_id, int count)
{
    return init_process_semaphore(sem_id, count, 0);
}

int get_semaphore_value(int sem_id, int number)
{
    int semval = semctl(sem_id, number, GETVAL, 0);
    if (semval < 0)
    {
        perror("semctl failed: ");
    }
    return semval;
}

int change_semaphore_value(int sem_id, int number, int change)
{
    struct sembuf buffer;
    buffer.sem_num = number;
    buffer.sem_op = change;
    buffer.sem_flg = 0;

    if (semop(sem_id, &buffer, 1) < 0)
    {
        perror("semop failed: ");
        return false;
    }
    return true;
}

int release_process_semaphore(int sem_id)
{
    if (semctl(sem_id, 0, IPC_RMID) < 0)
    {
        perror("semctl failed: ");
        return false;
    }
    return true;
}
#else
//Windows -> Not yet implemented
#include <stdio.h>
#include <stdlib.h>

#ifndef false
#define false 0
#endif

#ifndef true
#define true 1
#endif

#define NOTIMPLEMENTED() { \
    printf("FATAL ERROR: Shared memory isn't implemented on Windows yet.\n"); \
    fflush(stdout); \
}

str initialize_shared_memory(void)
{
    NOTIMPLEMENTED();
    return NULL;
}

str create_shared_memory(int id, size_t size, void **return_ptr)
{
    (void) id; (void) size; (void) return_ptr;
    NOTIMPLEMENTED();
    return NULL;
}

str release_shared_memory(void *ptr)
{
    (void) ptr;
    NOTIMPLEMENTED();
    return NULL;
}

int get_unique_shared_memory_id(int offset)
{
    (void) offset;
    NOTIMPLEMENTED();
    return -1;
}

str get_shared_memory(int id, size_t size, void **return_ptr)
{
    (void) id; (void) size; (void) return_ptr;
    NOTIMPLEMENTED();
    return NULL;
}


int create_process_semaphore(int id, int count)
{
    (void) id; (void) count;
    NOTIMPLEMENTED();
    return -1;
}

int get_process_semaphore(int id, int count)
{
    (void) id; (void) count;
    NOTIMPLEMENTED();
    return -1;
}

int get_semaphore_value(int sem_id, int number)
{
    (void) sem_id; (void) number;
    NOTIMPLEMENTED();
    return -1;
}

int change_semaphore_value(int sem_id, int number, int change)
{
    (void) sem_id; (void) number; (void) change;
    NOTIMPLEMENTED();
    return false;
}

int release_process_semaphore(int sem_id)
{
    (void) sem_id;
    NOTIMPLEMENTED();
    return false;
}
#endif
