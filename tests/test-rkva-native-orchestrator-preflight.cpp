#include "../tools/rkva-native-key-inspect/native-key-inspect.h"
#include "../tools/server/server-rkva-managed-orchestrator.h"
#undef NDEBUG
#include <cassert>
#include <filesystem>
#include <iostream>
#include <unistd.h>
namespace fs=std::filesystem;
namespace hj=rkva_host_journal;
namespace mr=rkva_managed_registry;
namespace mo=rkva_managed_orchestrator;
int main(int argc,char ** argv) {
    assert(argc==3);
    char tmp[]="/tmp/native-preflight-test-XXXXXX";char * root=mkdtemp(tmp);assert(root);
    const fs::path base(root),host=base/"host",journal_dir=base/"journal",peer_dir=base/"peer",objects=base/"empty-objects";
    for(const auto & d:{host,journal_dir,peer_dir,objects}) assert(fs::create_directory(d));
    const std::string owner(64,'a');hj::journal journal(journal_dir.string(),owner);mr::registry peer(peer_dir.string(),owner);std::string error;
    assert(journal.initialize(error) && peer.initialize(error));int verified=0;
    mo::source_preflight verify=[&](int fd,std::string & key,uint64_t & bytes,std::string & err){
        rkva_native_key_inspect::result result;
        if(!rkva_native_key_inspect::inspect_fd(fd,result,err)) return false;
        key=result.native_key_sha256;bytes=result.whole_file_bytes;++verified;return true;
    };
    mo::orchestrator flow(journal,peer,host.string(),objects.string(),8ull<<30,verify);
    const std::string keys[]={"68b109a4d14bbd26d7937cc9f8268a6bdb706584239531c440762857036d82c5",
                              "4d5532221ee86514043d1cabb790afb0b09e757ff11b824931b0de3bb0b24fc3"};
    for(int i=0;i<2;i++) {
        const std::string tx(64,char('1'+i)),ref(64,char('3'+i));
        assert(!flow.publish(tx,ref,keys[i],argv[i+1],error));
        assert(error.find("peer object")!=std::string::npos || error.find("managed file unavailable")!=std::string::npos);
        assert(verified==i+1 && !fs::exists(host/(tx+".managed")));
    }
    hj::snapshot host_records;mr::snapshot peer_records;
    assert(journal.inspect(host_records,error) && host_records.transactions.size()==2);
    for(const auto & r:host_records.transactions) assert(r.state==hj::phase::prepared);
    assert(peer.inspect(peer_records,error) && peer_records.transactions.empty());
    assert(fs::is_empty(objects));fs::remove_all(base);
    std::cout<<"native host preflight passes; absent peer fails closed before publication/commit: PASS\n";
}
