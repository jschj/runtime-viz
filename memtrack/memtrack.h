/*
    This is the common header for the library memtrack. The user can use the API to track a
    previously allocated buffer (malloc).
 */


// user API
extern void TRACK_BUFFER(void *location, const char *name);
