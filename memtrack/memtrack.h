/*
    This is the common header for the library memtrack. The user can use the API to track a
    previously allocated buffer (malloc).
 */


// user API
template <class T>
extern void TRACK_BUFFER(T *location, const char *name);
