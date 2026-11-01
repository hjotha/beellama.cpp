#include "rkva-managed-store-authority.h"
#include <system_error>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cwctype>
#include <algorithm>
#endif
#if defined(__linux__)
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
namespace rkva_managed_store {
struct lease::impl {
    std::filesystem::path parent;mutable std::recursive_mutex mutex;
#if defined(__linux__)
    int directory_fd=-1,lock_fd=-1;pid_t owner=0;dev_t directory_dev=0,lock_dev=0;ino_t directory_ino=0,lock_ino=0;
    ~impl(){if(lock_fd>=0)::close(lock_fd);if(directory_fd>=0)::close(directory_fd);}
#elif defined(_WIN32)
    HANDLE directory_handle=INVALID_HANDLE_VALUE,lock_handle=INVALID_HANDLE_VALUE;
    DWORD owner=0; OVERLAPPED offset{}; bool locked=false;
    BY_HANDLE_FILE_INFORMATION directory_id{},lock_id{};
    ~impl(){if(lock_handle!=INVALID_HANDLE_VALUE){if(locked)UnlockFileEx(lock_handle,0,1,0,&offset);CloseHandle(lock_handle);}if(directory_handle!=INVALID_HANDLE_VALUE)CloseHandle(directory_handle);}
#endif
};
#if defined(_WIN32)
namespace {
HANDLE open_safe(const std::filesystem::path & path,DWORD access,DWORD disposition,DWORD flags) {
    CREATEFILE2_EXTENDED_PARAMETERS p{};p.dwSize=sizeof(p);p.dwFileAttributes=FILE_ATTRIBUTE_NORMAL;
    p.dwFileFlags=flags|FILE_FLAG_OPEN_REPARSE_POINT;
    return CreateFile2(path.c_str(),access,FILE_SHARE_READ|FILE_SHARE_WRITE,disposition,&p);
}
bool info(HANDLE h,BY_HANDLE_FILE_INFORMATION & i,bool dir) {
    if(h==INVALID_HANDLE_VALUE || GetFileType(h)!=FILE_TYPE_DISK || !GetFileInformationByHandle(h,&i))return false;
    if(i.dwFileAttributes&FILE_ATTRIBUTE_REPARSE_POINT)return false;
    return dir ? (i.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)!=0 :
        !(i.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY) && i.nNumberOfLinks==1;
}
bool same(const BY_HANDLE_FILE_INFORMATION & a,const BY_HANDLE_FILE_INFORMATION & b) {
    return a.dwVolumeSerialNumber==b.dwVolumeSerialNumber && a.nFileIndexHigh==b.nFileIndexHigh && a.nFileIndexLow==b.nFileIndexLow;
}
}
#endif
lease::lease():data_(std::make_unique<impl>()){}
lease::~lease()=default;
bool lease::owns(const std::shared_ptr<lease> & candidate) const {
    const auto self=weak_from_this();
    return candidate && candidate.get()==this && !self.expired() && !self.owner_before(candidate) && !candidate.owner_before(self);
}
const std::filesystem::path & lease::directory()const{return data_->parent;}
std::shared_ptr<lease> lease::try_acquire(const std::filesystem::path & parent,std::string & error){
    error.clear();
#if defined(__linux__)
    try {
        std::error_code ec;const auto resolved=std::filesystem::canonical(parent,ec);
        if(ec){error="store parent unavailable";return {};}
        for(const auto & p:resolved)if(p=="rkva-snapshots-v1"||p=="rkva-snapshots-v2"){error="legacy namespace forbidden";return {};}
        auto out=std::shared_ptr<lease>(new lease());auto & d=*out->data_;d.parent=resolved;d.owner=::getpid();
        d.directory_fd=::open(parent.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW);struct stat dir{};
        if(d.directory_fd<0 || ::fstat(d.directory_fd,&dir) || !S_ISDIR(dir.st_mode) || dir.st_uid!=::geteuid() || (dir.st_mode&0022)){
            error="store must be a private owned directory";return {};}
        d.directory_dev=dir.st_dev;d.directory_ino=dir.st_ino;
        d.lock_fd=::openat(d.directory_fd,".rkva-managed-store.lock",O_RDWR|O_CREAT|O_CLOEXEC|O_NOFOLLOW,0600);struct stat lock{};
        if(d.lock_fd<0 || ::fstat(d.lock_fd,&lock) || !S_ISREG(lock.st_mode) || lock.st_nlink!=1 ||
                lock.st_uid!=::geteuid() || (lock.st_mode&0022)){
            error="invalid managed store lock";return {};}
        d.lock_dev=lock.st_dev;d.lock_ino=lock.st_ino;
        if(::flock(d.lock_fd,LOCK_EX|LOCK_NB)){error="managed store busy";return {};}
        if(!out->valid_for(resolved)){error="store authority changed during acquisition";return {};}
        return out;
    } catch(const std::exception & e){error=e.what();return {};}
#elif defined(_WIN32)
    try {
        if(!parent.is_absolute()){error="absolute private store parent required";return {};}
        auto resolved=parent.lexically_normal();
        for(const auto & part:resolved){auto p=part.wstring();std::transform(p.begin(),p.end(),p.begin(),[](wchar_t c){return wchar_t(std::towlower(c));});
            if(p==L"rkva-snapshots-v1"||p==L"rkva-snapshots-v2"){error="legacy namespace forbidden";return {};}}
        auto out=std::shared_ptr<lease>(new lease());auto & d=*out->data_;d.parent=resolved;d.owner=GetCurrentProcessId();
        d.directory_handle=open_safe(resolved,FILE_READ_ATTRIBUTES,OPEN_EXISTING,FILE_FLAG_BACKUP_SEMANTICS);
        if(!info(d.directory_handle,d.directory_id,true)){error="private store directory unavailable or reparse point";return {};}
        d.lock_handle=open_safe(resolved/L".rkva-managed-store.lock",GENERIC_READ|GENERIC_WRITE,OPEN_ALWAYS,0);
        if(!info(d.lock_handle,d.lock_id,false)){error="invalid managed store lock";return {};}
        if(!LockFileEx(d.lock_handle,LOCKFILE_EXCLUSIVE_LOCK|LOCKFILE_FAIL_IMMEDIATELY,0,1,0,&d.offset)){error="managed store busy";return {};}
        d.locked=true;
        if(!out->valid_for(resolved)){error="store authority changed during acquisition";return {};}
        return out;
    }catch(const std::exception & e){error=e.what();return {};}
#else
    (void)parent;error="managed store authority unsupported";return {};
#endif
}
std::unique_lock<std::recursive_mutex> lease::enter() const {
#if defined(__linux__)
    if(!data_ || data_->owner!=::getpid() || data_->lock_fd<0)return {};
    return std::unique_lock<std::recursive_mutex>(data_->mutex);
#elif defined(_WIN32)
    if(!data_ || data_->owner!=GetCurrentProcessId() || data_->lock_handle==INVALID_HANDLE_VALUE)return {};
    return std::unique_lock<std::recursive_mutex>(data_->mutex);
#else
    return {};
#endif
}
bool lease::valid_for(const std::filesystem::path & parent) const {
#if defined(__linux__)
    if(!data_ || data_->owner!=::getpid() || data_->directory_fd<0 || data_->lock_fd<0)return false;
    struct stat opened{},path{},held{},current{};
    if(::fstat(data_->directory_fd,&opened)||::lstat(parent.c_str(),&path)||!S_ISDIR(path.st_mode)||
       ::fstat(data_->lock_fd,&held)||::fstatat(data_->directory_fd,".rkva-managed-store.lock",&current,AT_SYMLINK_NOFOLLOW))return false;
    return opened.st_uid==::geteuid() && !(opened.st_mode&0022) && path.st_uid==opened.st_uid && !(path.st_mode&0022) &&
       S_ISREG(held.st_mode)&&S_ISREG(current.st_mode)&&held.st_nlink==1&&current.st_nlink==1&&
       held.st_uid==::geteuid() && current.st_uid==held.st_uid && !(held.st_mode&0022) && !(current.st_mode&0022) &&
       opened.st_dev==data_->directory_dev&&opened.st_ino==data_->directory_ino&&
       path.st_dev==opened.st_dev&&path.st_ino==opened.st_ino&&
       held.st_dev==data_->lock_dev&&held.st_ino==data_->lock_ino&&current.st_dev==held.st_dev&&current.st_ino==held.st_ino;
#elif defined(_WIN32)
    if(!data_ || !data_->locked || data_->owner!=GetCurrentProcessId() || data_->directory_handle==INVALID_HANDLE_VALUE || data_->lock_handle==INVALID_HANDLE_VALUE)return false;
    BY_HANDLE_FILE_INFORMATION held_dir{},held_lock{},path_dir{},path_lock{};
    HANDLE dh=open_safe(parent,FILE_READ_ATTRIBUTES,OPEN_EXISTING,FILE_FLAG_BACKUP_SEMANTICS);
    HANDLE lh=open_safe(parent/L".rkva-managed-store.lock",FILE_READ_ATTRIBUTES,OPEN_EXISTING,0);
    const bool valid=info(data_->directory_handle,held_dir,true)&&info(data_->lock_handle,held_lock,false)&&
        info(dh,path_dir,true)&&info(lh,path_lock,false)&&same(data_->directory_id,held_dir)&&same(held_dir,path_dir)&&
        same(data_->lock_id,held_lock)&&same(held_lock,path_lock);
    if(dh!=INVALID_HANDLE_VALUE)CloseHandle(dh);
    if(lh!=INVALID_HANDLE_VALUE)CloseHandle(lh);
    return valid;
#else
    (void)parent;return false;
#endif
}
}
