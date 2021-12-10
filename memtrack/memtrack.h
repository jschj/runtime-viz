/*
    This is the common header for the library memtrack. The user can use the API to track a
    previously allocated buffer (malloc).
 */


// user API
extern void TRACK_BUFFER(void *location, const char *name);

//template <class ElemType = float>
//extern void _TRACK_BUFFER(ElemType *location, const char *name);