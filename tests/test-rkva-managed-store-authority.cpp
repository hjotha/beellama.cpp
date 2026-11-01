#include "../ggml/include/rkva-managed-store-authority.h"
#include "../ggml/include/rkva-anchored-sqlite.h"
#include "../tools/xbox/uwp/rkva/rkva-managed-object-v3.h"
#undef NDEBUG
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <thread>
#include <sys/wait.h>
#include <unistd.h>
namespace fs=std::filesystem;namespace authority=rkva_managed_store;namespace db=rkva_anchored_sqlite;namespace v3=rkva_managed_object_v3;
static std::string contents(const fs::path & p){std::ifstream f(p);return std::string((std::istreambuf_iterator<char>(f)),{});}
struct fake_pin:v3::pin_proof {
 int calls=0;
 bool pending(const v3::pin &,const v3::expected &,const std::shared_ptr<authority::lease> &) override {++calls;return true;}
 bool pinned(const v3::pin &,const v3::expected &,const std::shared_ptr<authority::lease> &) override {++calls;return true;}
};
struct fake_object:db::object_proof {
 int calls=0;
 bool verify(db::operation,const db::record &,const std::shared_ptr<authority::lease> &) override {++calls;return true;}
};
struct fake_disk:v3::durable_filesystem {
 mutable int calls=0;
 fake_disk(std::weak_ptr<authority::lease> token,fs::path root):durable_filesystem(std::move(token),std::move(root)) {}
 bool qualified() const override { ++calls;return true; }
 bool sync_file(const fs::path &,std::string &) override {++calls;return true;}
 bool sync_directory(const fs::path &,std::string &) override {++calls;return true;}
 v3::move_result rename_noreplace_and_sync(const fs::path &,const fs::path &,std::string &) override {++calls;return v3::move_result::durable;}
};
int main(int argc,char ** argv){
 std::string error;
 if(argc==3 && std::string(argv[1])=="--try"){
  auto token=authority::lease::try_acquire(argv[2],error);return token?0:3;
 }
 char tmp[]="/tmp/rkva-common-authority-XXXXXX";char * name=mkdtemp(tmp);assert(name);fs::path parent(name),other=parent/"other",objects=parent/"rkva-snapshots-v3";
 assert(fs::create_directory(other)&&fs::create_directory(objects));{std::ofstream f(objects/"sentinel");f<<"preserve";}
 db::id owner{},epoch{};db::secret key{};owner.fill(1);epoch.fill(2);key.fill(3);
 db::backend sqlite(parent.string(),owner,epoch,key);assert(!sqlite.provision({},error));assert(!fs::exists(parent/"rkva-managed-v3"));
 auto token=authority::lease::try_acquire(parent,error);assert(token && token->valid_for(parent));
 // Multiple threads sharing the same open-file-description still serialize
 // through the token's operation mutex, rather than relying on flock alone.
 {
  auto held=token->enter();assert(held.owns_lock());std::promise<void> started,finished;
  auto a=started.get_future(),b=finished.get_future();
  std::thread thread([&](){started.set_value();auto next=token->enter();assert(next.owns_lock());finished.set_value();});
  a.wait();assert(b.wait_for(std::chrono::milliseconds(50))==std::future_status::timeout);
  held.unlock();b.wait();thread.join();
 }
 auto unrelated=authority::lease::try_acquire(other,error);assert(unrelated);
 assert(!sqlite.provision(unrelated,error));assert(!fs::exists(parent/"rkva-managed-v3"));
 assert(sqlite.provision(token,error));db::state zero{};assert(sqlite.inspect(token,zero,error));
 db::state untouched{};untouched.version=999;assert(!sqlite.inspect({},untouched,error)&&untouched.version==999);
 v3::pin pin{};pin.owner.fill(1);pin.transaction.fill(2);pin.reference.fill(3);pin.key.fill(4);v3::expected geometry{};
 fake_disk disk(token,objects),other_disk(unrelated,objects),wrong_root(token,other/"rkva-snapshots-v3");
 auto rejected=v3::write(objects,{},pin,geometry,nullptr,nullptr,nullptr,nullptr);
 assert(rejected.state==v3::publication::rejected && rejected.error.find("filesystem/store authority")!=std::string::npos);
 rejected=v3::write(objects,unrelated,pin,geometry,nullptr,nullptr,&disk,nullptr);
 assert(rejected.state==v3::publication::rejected && rejected.error.find("lease mismatch")!=std::string::npos);
 rejected=v3::write(objects,token,pin,geometry,nullptr,nullptr,&other_disk,nullptr);
 assert(rejected.state==v3::publication::rejected && rejected.error.find("lease mismatch")!=std::string::npos);
 auto counterfeit=std::shared_ptr<authority::lease>(token.get(),[](authority::lease *){});
 fake_disk forged(counterfeit,objects);
 rejected=v3::write(objects,token,pin,geometry,nullptr,nullptr,&forged,nullptr);
 assert(rejected.state==v3::publication::rejected && rejected.error.find("lease mismatch")!=std::string::npos);
 rejected=v3::write(objects,token,pin,geometry,nullptr,nullptr,&wrong_root,nullptr);
 assert(rejected.state==v3::publication::rejected && rejected.error.find("root mismatch")!=std::string::npos);
 fs::path third=parent/"third";assert(fs::create_directory(third));auto temporary=authority::lease::try_acquire(third,error);assert(temporary);
 fake_disk expired(temporary,objects);temporary.reset();assert(expired.authority().expired());
 rejected=v3::write(objects,token,pin,geometry,nullptr,nullptr,&expired,nullptr);
 assert(rejected.state==v3::publication::rejected && rejected.error.find("expired")!=std::string::npos);
 assert(disk.calls==0 && other_disk.calls==0 && forged.calls==0 && wrong_root.calls==0 && expired.calls==0);
 fake_pin false_claim;
 rejected=v3::write(objects,token,pin,geometry,&false_claim,nullptr,&disk,nullptr);
 assert(rejected.state==v3::publication::rejected && rejected.error.find("concrete authenticated pin")!=std::string::npos && false_claim.calls==0);
 std::string digest="unchanged";assert(!v3::verify(objects,{},pin,geometry,nullptr,digest,error)&&digest=="unchanged");
 assert(!v3::verify(objects,token,pin,geometry,&false_claim,digest,error)&&false_claim.calls==0 && digest=="unchanged");
 assert(contents(objects/"sentinel")=="preserve");
 // A different process cannot acquire the same concrete store token while it
 // is held. The inherited descriptor is closed during exec (O_CLOEXEC).
 pid_t child=fork();assert(child>=0);if(child==0){execl(argv[0],argv[0],"--try",parent.c_str(),(char*)nullptr);_exit(127);}
 int status=0;assert(waitpid(child,&status,0)==child && WIFEXITED(status) && WEXITSTATUS(status)==3);
 // A forked child may not reuse the parent's inherited C++ token either.
 child=fork();assert(child>=0);if(child==0){
  auto inherited=v3::write(objects,token,pin,geometry,&false_claim,nullptr,&disk,nullptr);
  _exit(!token->valid_for(parent) && inherited.state==v3::publication::rejected &&
        inherited.error.find("concrete store authority")!=std::string::npos && disk.calls==0 ? 0:5);
 }
 assert(waitpid(child,&status,0)==child && WIFEXITED(status) && WEXITSTATUS(status)==0);
 // The same token is required from challenge issuance through apply.
 db::anchor anchor;anchor.owner=owner;anchor.epoch=epoch;anchor.accepted=zero;
 assert(sqlite.challenge(token,anchor.challenge,error)&&db::sign_anchor(key,anchor));
 db::command command{db::operation::hello_claim};command.nonce.fill(8);db::state output{};
 assert(sqlite.apply({},&anchor,command,output,error)==db::result::unavailable);
 assert(sqlite.apply(token,&anchor,command,output,error)==db::result::ok && output.version==zero.version+1);
 // A true-returning object fixture cannot authorize a fake durable pin.
 db::anchor pending_anchor;pending_anchor.owner=owner;pending_anchor.epoch=epoch;pending_anchor.accepted=output;
 assert(sqlite.challenge(token,pending_anchor.challenge,error)&&db::sign_anchor(key,pending_anchor));
 db::command pending{db::operation::prepare};pending.tx.fill(2);pending.reference.fill(3);pending.key.fill(4);pending.nonce.fill(11);
 fake_object unsupported;assert(sqlite.apply(token,&pending_anchor,pending,output,error,nullptr,nullptr,&unsupported)==db::result::denied && unsupported.calls==0);
 // Rollback anchor must not publish state or disturb any object.
 db::anchor stale;stale.owner=owner;stale.epoch=epoch;stale.accepted=zero;
 assert(sqlite.challenge(token,stale.challenge,error)&&db::sign_anchor(key,stale));command.nonce.fill(9);
 assert(sqlite.apply(token,&stale,command,output,error)==db::result::stale_anchor);
 db::state latest{};assert(sqlite.inspect(token,latest,error)&&latest.version==1 && contents(objects/"sentinel")=="preserve");
 // After release and reacquire, an outstanding challenge from the old token is
 // invalid, even when the new token names the same directory.
 db::anchor old;old.owner=owner;old.epoch=epoch;old.accepted=latest;assert(sqlite.challenge(token,old.challenge,error)&&db::sign_anchor(key,old));
 std::weak_ptr<authority::lease> weak=token;token.reset();assert(weak.expired());
 token=authority::lease::try_acquire(parent,error);assert(token);command.nonce.fill(10);
 rejected=v3::write(objects,token,pin,geometry,&false_claim,nullptr,&disk,nullptr);
 assert(rejected.state==v3::publication::rejected && rejected.error.find("expired")!=std::string::npos && disk.calls==0);
 assert(sqlite.apply(token,&old,command,output,error)==db::result::invalid_anchor);
 assert(sqlite.inspect(token,latest,error)&&latest.version==1);
 // Replacement of the lock pathname invalidates the old authority before
 // SQLite or the writer can proceed, while existing data remains intact.
 fs::rename(parent/".rkva-managed-store.lock",parent/"old-lock");
 {std::ofstream f(parent/".rkva-managed-store.lock");f<<"new";}
 assert(!token->valid_for(parent));assert(!sqlite.inspect(token,latest,error));
 fake_disk disk_reacquired(token,objects);
 assert(v3::write(objects,token,pin,geometry,&false_claim,nullptr,&disk_reacquired,nullptr).state==v3::publication::rejected);
 assert(false_claim.calls==0 && contents(objects/"sentinel")=="preserve");
 token.reset();child=fork();assert(child>=0);if(child==0){execl(argv[0],argv[0],"--try",parent.c_str(),(char*)nullptr);_exit(127);}
 assert(waitpid(child,&status,0)==child && WIFEXITED(status) && WEXITSTATUS(status)==0);
 unrelated.reset();fs::remove_all(parent);
 std::cout<<"shared POSIX store authority: missing/wrong/released lease, process exclusion, stale anchor, preservation: PASS\n";
}
