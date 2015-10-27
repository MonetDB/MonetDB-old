
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
#include <time.h>

static lng *shm_memory_ids;
static void **shm_ptrs;
static int shm_unique_id = 1;
static int shm_current_id = 0;
static int shm_max_id = 32;
static int shm_is_initialized = false;
static MT_Lock release_memory_lock;
static key_t base_key = 800000000;

#define SHM_SHARED 1
#define SHM_MEMMAP 2
#define SHM_EITHER 3

static int memtype = SHM_SHARED;


str init_shared_memory(int id, size_t size, void **return_ptr, int flags, bool reg, lng *return_shmid);
void store_shared_memory(lng memory_id, void *ptr);
str release_shared_memory_id(int memory_id, void *ptr);

str init_mmap_memory(int id, size_t size, void **return_ptr, int flags, bool reg, lng *return_shmid);
str release_mmap_memory(void *ptr, size_t size);

str init_process_semaphore(int id, int count, int flags, int *semid);

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

str init_mmap_memory(int id, size_t size, void **return_ptr, int flags, bool reg, lng *return_shmid)
{   
    char address[100];
    void *ptr;
    int fd, result;
    // TODO: memmap shouldn't be in tmp directory
    // TODO: we should just use GDKmmap, try to get that to work
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
    if (reg) store_shared_memory(size, ptr);
    if (return_ptr != NULL) *return_ptr = ptr;
    if (return_shmid != NULL) *return_shmid = id;
    return MAL_SUCCEED;
}

str release_mmap_memory(void *ptr, size_t size)
{
    int ret;
    // TODO: Actually delete files on disk
    ret = munmap(ptr, size);
    if (ret != 0) {
        char *err = strerror(errno);
        errno = 0;
        return createException(MAL, "shared_memory.release_mmap_memory", "Failure in munmap(%p,%zu): %s", ptr, size, err);
    }
    return MAL_SUCCEED;
}

str release_shared_memory_ptr(void *ptr)
{
    if (shmdt(ptr) == -1)
    {
        char *err = strerror(errno);
        errno = 0;
        return createException(MAL, "shared_memory.release", "Error calling shmdt(ptr:%p): %s", ptr, err);
    }
    return MAL_SUCCEED;
}

str create_shared_memory(int id, size_t size, bool reg, void **return_ptr, lng *return_shmid)
{
    char *shared, *mmap;
    if (memtype == SHM_SHARED)
    {
        return init_shared_memory(id, size, return_ptr, IPC_CREAT, reg, return_shmid);
    }
    if (memtype == SHM_MEMMAP)
    {
        return init_mmap_memory(id, size, return_ptr, O_CREAT, reg, return_shmid);
    }
    if ((shared = init_shared_memory(id, size, return_ptr, IPC_CREAT, reg, return_shmid)) == MAL_SUCCEED) return MAL_SUCCEED;
    if ((mmap = init_mmap_memory(id, size, return_ptr, O_CREAT, reg, return_shmid)) == MAL_SUCCEED) return MAL_SUCCEED;
    return createException(MAL, "shared_memory.release_mmap_memory", "Failed to create shared memory or mmap space.\nshared memory error: %s\nmmap error: %s", shared, mmap);
}

str get_shared_memory(int id, size_t size, bool reg, void **return_ptr, lng *return_shmid)
{
    char *shared, *mmap;
    if (memtype == SHM_SHARED)
    {
        return init_shared_memory(id, size, return_ptr, 0, reg, return_shmid);
    }
    if (memtype == SHM_MEMMAP)
    {
        return init_mmap_memory(id, size, return_ptr, 0, reg, return_shmid);
    }
    if ((shared = init_shared_memory(id, size, return_ptr, 0, reg, return_shmid)) == MAL_SUCCEED) return MAL_SUCCEED;
    if ((mmap = init_mmap_memory(id, size, return_ptr, 0, reg, return_shmid)) == MAL_SUCCEED) return MAL_SUCCEED;
    return createException(MAL, "shared_memory.release_mmap_memory", "Failed to get shared memory or mmap space.\nshared memory error: %s\nmmap error: %s", shared, mmap);
}

str ftok_enhanced(int id, key_t *return_key);
str ftok_enhanced(int id, key_t *return_key)
{
    *return_key = base_key + id;
    return MAL_SUCCEED;
}

str init_shared_memory(int id, size_t size, void **return_ptr, int flags, bool reg, lng *return_shmid)
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

    if (reg) {
        //check if the shared memory segment is already created, if it is we do not need to add it to the table and can simply return the pointer
        for(i = 0; i < shm_current_id; i++)
        {
            if (shm_memory_ids[i] == shmid)
            {
                if (return_ptr != NULL) *return_ptr = shm_ptrs[i];
                return MAL_SUCCEED;
            }
        }
    }


	ptr = shmat(shmid, NULL, 0);
    if (ptr == (void*)-1)
    {
        char *err = strerror(errno);
        errno = 0;
        return createException(MAL, "shared_memory.get", "Error calling shmat(id:%lld,NULL,0): %s", shmid, err);
    }

    if (reg) store_shared_memory(shmid, ptr);
    if (return_shmid != NULL) *return_shmid = shmid;
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
    return release_shared_memory_shmid(memory_id, ptr);
}

str release_shared_memory_shmid(int memory_id, void *ptr)
{
    assert(memory_id);

    if (memtype == SHM_SHARED)
    {
        return release_shared_memory_id(memory_id, ptr);
    }
    if (memtype == SHM_MEMMAP)
    {
        return release_mmap_memory(ptr, memory_id);
    }
    if (release_shared_memory_id(memory_id, ptr) == MAL_SUCCEED) return MAL_SUCCEED;
    if (release_mmap_memory(ptr, memory_id) == MAL_SUCCEED) return MAL_SUCCEED;
    return createException(MAL, "shared_memory.release", "Failed to release shared memory.");
}

str release_shared_memory_id(int memory_id, void *ptr)
{   
    if (shmdt(ptr) == -1)
    {
        char *err = strerror(errno);
        errno = 0;
        return createException(MAL, "shared_memory.release", "Error calling shmdt(ptr:%p): %s", ptr, err);
    }
	if (shmctl(memory_id, IPC_RMID, NULL) == -1)
	{
        char *err = strerror(errno);
        errno = 0;
        return createException(MAL, "shared_memory.release", "Error calling shmctl(id:%d,IPC_RMID,NULL): %s", memory_id, err);
	}
	return MAL_SUCCEED;
}

str init_process_semaphore(int id, int count, int flags, int *semid)
{
    str msg = MAL_SUCCEED;
    int key;
    msg = ftok_enhanced(id, &key);
    if (msg != MAL_SUCCEED) {
        return msg;
    }
    *semid = semget(key, count, flags | 0666);
    if (*semid < 0) {
        char *err = strerror(errno);
        errno = 0;
        return createException(MAL, "semaphore.init", "Error calling semget(key:%d,nsems:%d,semflg:%d): %s", key, count, flags | 0666, err);
    }
    return msg;
}

str create_process_semaphore(int id, int count, int *semid)
{
    return init_process_semaphore(id, count, IPC_CREAT, semid);
}

str get_process_semaphore(int sem_id, int count, int *semid)
{
    return init_process_semaphore(sem_id, count, 0, semid);
}

str get_semaphore_value(int sem_id, int number, int *semval)
{
    *semval = semctl(sem_id, number, GETVAL, 0);
    if (*semval < 0)
    {
        char *err = strerror(errno);
        errno = 0;
        return createException(MAL, "semaphore.init", "Error calling semctl(semid:%d,semnum:%d,cmd:%d,param:0): %s", sem_id, number, GETVAL, err);
    }
    return MAL_SUCCEED;
}

str change_semaphore_value(int sem_id, int number, int change)
{
    str msg = MAL_SUCCEED;
    struct sembuf buffer;
    buffer.sem_num = number;
    buffer.sem_op = change;
    buffer.sem_flg = 0;

    if (semop(sem_id, &buffer, 1) < 0)
    {
        char *err = strerror(errno);
        errno = 0;
        return createException(MAL, "semaphore.init", "Error calling semop(semid:%d, sops: { sem_num:%d, sem_op:%d, sem_flag: %d }, nsops:1): %s", sem_id, number, change, 0, err);
    }
    return msg;
}

str change_semaphore_value_timeout(int sem_id, int number, int change, int timeout_mseconds, bool *succeed)
{
    str msg = MAL_SUCCEED;
    struct timespec timeout;
    struct sembuf buffer;
    buffer.sem_num = number;
    buffer.sem_op = change;
    buffer.sem_flg = 0;
    *succeed = false;

    timeout.tv_sec = (timeout_mseconds / 1000);
    timeout.tv_nsec = (timeout_mseconds % 1000) * 1000;

    if (semtimedop(sem_id, &buffer, 1, &timeout) < 0)
    {
        if (errno == EAGAIN) {
            errno = 0;
            return MAL_SUCCEED;
        } else {
            char *err = strerror(errno);
            errno = 0;
            return createException(MAL, "semaphore.init", "Error calling semtimedop(semid:%d, sops: { sem_num:%d, sem_op:%d, sem_flag: %d }, nsops:1): %s", sem_id, number, change, 0, err);
        }
    }
    *succeed = true;
    return msg;
}

str release_process_semaphore(int sem_id)
{
    if (semctl(sem_id, 0, IPC_RMID) < 0)
    {
        char *err = strerror(errno);
        errno = 0;
        return createException(MAL, "semaphore.init", "Error calling semctl(%d, 0, IPC_RMID): %s", sem_id, err);
    }
    return MAL_SUCCEED;
}

#endif
