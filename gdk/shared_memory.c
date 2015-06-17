
#include "shared_memory.h"

#ifndef false
#define false 0
#endif

#ifndef true
#define true 1
#endif

#ifndef _WIN32

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

#include "monetdb_config.h"
#include "gdk.h"

static int *shm_memory_ids;
static void **shm_ptrs;
static int shm_unique_id = 1;
static int shm_current_id = 0;
static int shm_max_id = 32;
static int shm_is_initialized = false;
static char shm_keystring[] = ".";

void *init_shared_memory(int id, size_t size, int flags);
void store_shared_memory(int memory_id, void *ptr);
int release_shared_memory_id(int memory_id, void *ptr);

int init_process_semaphore(int id, int count, int flags);

void initialize_shared_memory(void)
{
	if (shm_is_initialized) return;

	shm_ptrs = malloc(shm_max_id * sizeof(void*));
	shm_memory_ids = malloc(shm_max_id * sizeof(int));
	shm_current_id = 0;
	shm_max_id = 32;
	shm_unique_id = 2;

	shm_is_initialized = true;
}

void store_shared_memory(int memory_id, void *ptr)
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
		int *new_memory_ids = malloc(shm_max_id * 2 * sizeof(int));

		memcpy(new_ptrs, shm_ptrs, sizeof(void*) * shm_max_id);
		memcpy(new_memory_ids, shm_memory_ids, sizeof(int) * shm_max_id);

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

void* create_shared_memory(int id, size_t size)
{
	return init_shared_memory(id, size, IPC_CREAT);
}

void *get_shared_memory(int id, size_t size)
{
	return init_shared_memory(id, size, 0);
}

void *init_shared_memory(int id, size_t size, int flags)
{
    int shmid;
    void *ptr;
    int i;
	int key = ftok(shm_keystring, id);
    if (key == (key_t) -1)
    {
        perror("ftok");
        return NULL;
    }

	assert(shm_is_initialized);

	shmid = shmget(key, size, flags | 0666);
    if (shmid < 0)
    {
    	perror("shmget");
        return NULL;
    }

    //check if the shared memory segment is already created, if it is we do not need to add it to the table and can simply return the pointer
    for(i = 0; i < shm_current_id; i++)
    {
        if (shm_memory_ids[i] == shmid)
        {
            return shm_ptrs[i];
        }
    }

	ptr = shmat(shmid, NULL, 0);
    if (ptr == (void*)-1)
    {
    	perror("shmat");
        return NULL;
    }

    store_shared_memory(shmid, ptr);
    return ptr;
}

int release_shared_memory(void *ptr)
{
	int i = 0;
	int memory_id = -1;

    assert(shm_is_initialized);

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

	assert(memory_id);
	//actually release the memory at the given ID
	return release_shared_memory_id(memory_id, ptr);
}

int release_shared_memory_id(int memory_id, void *ptr)
{
	if (shmctl(memory_id, IPC_RMID, NULL) == -1)
	{
    	perror("shmctl");
        return false;
	}
	if (shmdt(ptr) == -1)
	{
    	perror("shmdt");
        return false;
	}
	return true;
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

#define NOTIMPLEMENTED() { \
    printf("FATAL ERROR: Shared memory isn't implemented on Windows yet.\n"); \
    fflush(stdout); \
}

void initialize_shared_memory(void)
{
    NOTIMPLEMENTED();
}

void* create_shared_memory(int id, size_t size)
{
    (void) id; (void) size;
    NOTIMPLEMENTED();
    return NULL;
}

int release_shared_memory(void *ptr)
{
    (void) ptr;
    NOTIMPLEMENTED();
    return false;
}

int get_unique_shared_memory_id(int offset)
{
    (void) offset;
    NOTIMPLEMENTED();
    return -1;
}  

void *get_shared_memory(int id, size_t size)
{
    (void) id; (void) size;
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
