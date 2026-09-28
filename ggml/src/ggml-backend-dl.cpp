#include "ggml-backend-dl.h"

#ifdef _WIN32

#if defined(WINAPI_FAMILY) && !WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)

// AppContainer (UWP/Xbox): SetErrorMode/SEM_FAILCRITICALERRORS are desktop-only
// and error dialogs cannot appear anyway; only packaged DLLs are loadable.
dl_handle * dl_load_library(const fs::path & path) {
    return LoadPackagedLibrary(path.wstring().c_str(), 0);
}

void * dl_get_sym(dl_handle * handle, const char * name) {
    return (void *) GetProcAddress(handle, name);
}

#else

dl_handle * dl_load_library(const fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    return handle;
}

void * dl_get_sym(dl_handle * handle, const char * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    void * p = (void *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    return p;
}

#endif // WINAPI_FAMILY app partition

const char * dl_error() {
    return "";
}

#else

dl_handle * dl_load_library(const fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    return handle;
}

void * dl_get_sym(dl_handle * handle, const char * name) {
    return dlsym(handle, name);
}

const char * dl_error() {
    const char *rslt = dlerror();
    return rslt != nullptr ? rslt : "";
}

#endif
