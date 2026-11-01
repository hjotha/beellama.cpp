#include "rkva-managed-object-v3.h"
#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <system_error>
namespace rkva_managed_object_v3 {
namespace {
namespace snap = rkva_native_snapshot;
void need(bool value,const std::string & error) { if(!value) throw std::runtime_error(error); }
bool nonzero(const rkva_snapshot::Digest & x) { return std::any_of(x.begin(),x.end(),[](uint8_t b){return b!=0;}); }
bool valid_pin(const pin & p) { return nonzero(p.owner)&&nonzero(p.transaction)&&nonzero(p.reference)&&nonzero(p.key); }
fs::file_status path_status(const fs::path & path) {
    std::error_code ec; const auto s=fs::symlink_status(path,ec);
    if(ec==std::errc::no_such_file_or_directory) return fs::file_status(fs::file_type::not_found);
    need(!ec,"object path inspection failed"); return s;
}
void validate_root(const fs::path & root) {
    need(!root.empty() && root.filename()=="rkva-snapshots-v3", "v3 namespace required");
    const auto s=path_status(root); need(fs::is_directory(s) && !fs::is_symlink(s),"invalid v3 store root");
    // Roots nested in legacy namespaces are never used, even if the final name
    // happens to be v3. Trusted path ancestors are still a caller obligation.
    for(const auto & part : root.lexically_normal())
        need(part!="rkva-snapshots-v1" && part!="rkva-snapshots-v2","legacy namespace forbidden");
}
void hold(pin_proof * proof,const pin & p,const expected & e,const std::shared_ptr<rkva_managed_store::lease> & store) {
    need(proof && proof->pending(p,e,store),"v3 durable pending pin unavailable");
}
void regular(const fs::path & path,uint64_t size) {
    const auto s=path_status(path); need(fs::is_regular_file(s) && !fs::is_symlink(s),"v3 object entry is not a regular file");
    std::error_code ec; const auto actual=fs::file_size(path,ec); need(!ec&&actual==size,"v3 object extent differs");
    const auto links=fs::hard_link_count(path,ec); need(!ec&&links==1,"v3 object file has unsafe hardlinks");
}
std::set<std::string> names(const fs::path & dir) {
    std::set<std::string> found; std::error_code ec;
    for(fs::directory_iterator it(dir,ec),end; !ec && it!=end; it.increment(ec)) {
        need(found.size()<snap::kMaxChunks+1,"v3 object entry limit exceeded");
        need(found.insert(it->path().filename().string()).second,"duplicate v3 object entry");
    }
    need(!ec,"v3 object listing failed"); return found;
}
std::vector<uint8_t> load_manifest(const fs::path & path) {
    const auto s=path_status(path); need(fs::is_regular_file(s)&&!fs::is_symlink(s),"invalid v3 manifest");
    std::error_code ec; auto n=fs::file_size(path,ec); need(!ec&&n<=snap::kMaxManifestBytes,"v3 manifest too large");
    regular(path,n); std::vector<uint8_t> bytes(size_t(n),uint8_t{});
    std::ifstream input(path,std::ios::binary);
    need(bool(input) && (bytes.empty() || bool(input.read(reinterpret_cast<char *>(bytes.data()),std::streamsize(bytes.size())))),
         "v3 manifest read failed");
    regular(path,n); return bytes;
}
void validate_directory(const fs::path & dir,const pin & p,const expected & e,std::string & digest) {
    const auto s=path_status(dir); need(fs::is_directory(s)&&!fs::is_symlink(s),"invalid v3 object directory");
    const auto bytes=load_manifest(dir/"manifest-v2.bin"); snap::Manifest m; std::string why;
    if (!snap::decode(bytes,m,&why)) throw std::runtime_error("v3 manifest decode failed: "+why);
    std::vector<uint8_t> canonical;
    need(snap::encode(m,canonical,&why) && canonical==bytes,"v3 manifest is noncanonical");
    if (!snap::validate(m,e.geometry,e.descriptor,e.rows,e.generation,p.key.data(),&why))
        throw std::runtime_error("v3 manifest identity mismatch: "+why);
    need(m.layers.size()==e.layers.size(),"v3 layer coverage count differs");
    for(size_t i=0;i<m.layers.size();i++) {
        const auto & a=m.layers[i]; const auto & b=e.layers[i];
        need(a.tail_begin==b.tail_begin && a.stage_begin==b.stage_begin && a.record_groups==b.record_groups && a.ready==b.ready,
             "v3 layer coverage differs");
    }
    std::set<std::string> expected_names{"manifest-v2.bin"};
    for(size_t i=0;i<m.chunks.size();i++) expected_names.insert("n-"+std::to_string(i)+".kv");
    need(names(dir)==expected_names,"v3 object has missing or unexpected entries");
    std::array<uint8_t,64*1024> buffer{};
    for(size_t i=0;i<m.chunks.size();i++) {
        const auto & c=m.chunks[i]; const auto path=dir/("n-"+std::to_string(i)+".kv");
        need(c.bytes && c.bytes<=snap::kChunkBytes,"v3 chunk extent invalid"); regular(path,c.bytes);
        std::ifstream input(path,std::ios::binary); need(bool(input),"v3 chunk cannot be read");
        rkva_snapshot::Sha256 hash; uint64_t remaining=c.bytes;
        while(remaining) { const size_t n=size_t(std::min<uint64_t>(remaining,buffer.size()));
            need(bool(input.read(reinterpret_cast<char *>(buffer.data()),std::streamsize(n))),"v3 chunk read failed");
            hash.update(buffer.data(),n); remaining-=n; }
        regular(path,c.bytes); need(hash.finish()==c.digest,"v3 chunk checksum mismatch");
    }
    need(names(dir)==expected_names && load_manifest(dir/"manifest-v2.bin")==bytes,"v3 object changed during verification");
    const auto hash=rkva_snapshot::digest(bytes.data(),bytes.size()); digest=rkva_snapshot::hex(hash.data());
}
} // namespace
bool verify(const fs::path & root,std::shared_ptr<rkva_managed_store::lease> store,const pin & p,const expected & e,pin_proof * proof,
            std::string & digest,std::string & error) {
    try { auto operation=store && store->owns(store)?store->enter():std::unique_lock<std::recursive_mutex>{};
        need(operation.owns_lock() && store->valid_for(root.parent_path()),"concrete store authority unavailable");
        validate_root(root); need(valid_pin(p),"invalid v3 pin");
        // A callback cannot establish an authenticated SQLite pin/metadata
        // binding. Keep object access closed until the concrete proof exists.
        need(false,"concrete authenticated pin and metadata proof unavailable");
        need(proof && proof->pinned(p,e,store),"v3 durable pin unavailable");
        std::string local; validate_directory(root/rkva_snapshot::hex(p.key.data()),p,e,local);
        need(store->valid_for(root.parent_path()) && proof->pinned(p,e,store),"v3 authority/pin lost");
        digest=std::move(local); error.clear(); return true;
    } catch(const std::exception & ex) { error=ex.what(); return false; }
      catch(...) { error="v3 verification failed"; return false; }
}
result write(const fs::path & root,std::shared_ptr<rkva_managed_store::lease> store,const pin & p,const expected & e,pin_proof * proof,quota_lease * quota,
             durable_filesystem * disk,source * input) {
    result r; bool move_started=false;
    try {
        // Compare shared ownership as well as the pointee: a separately owned
        // alias must not impersonate the concrete lease. No backend callbacks
        // or object-path I/O occur before these checks.
        need(store && disk,"v3 filesystem/store authority unavailable");
        const auto bound=disk->authority().lock();
        need(bound && bound.get()==store.get() && !bound.owner_before(store) && !store.owner_before(bound),
             "v3 filesystem/store lease mismatch or expired");
        need(!root.empty() && disk->root()==root.lexically_normal(),"v3 filesystem root mismatch");
        auto operation=store && store->owns(store)?store->enter():std::unique_lock<std::recursive_mutex>{};
        need(operation.owns_lock() && store->valid_for(root.parent_path()),"concrete store authority unavailable");
        validate_root(root); need(valid_pin(p),"invalid v3 pin");
        need(false,"concrete authenticated pin and metadata proof unavailable");
        hold(proof,p,e,store);
        need(quota && disk && input && disk->qualified(),"v3 durable backend or reservation unavailable");
        need(input->bound_to(p,e),"v3 source binding unavailable");
        snap::Manifest manifest; std::string why;
        if (!snap::build(e.geometry,e.descriptor,e.rows,e.generation,p.key.data(),e.layers,manifest,&why))
            throw std::runtime_error("v3 snapshot geometry rejected: "+why);
        std::vector<uint8_t> encoded;
        need(snap::encode(manifest,encoded,&why),"v3 manifest sizing failed");
        constexpr uint64_t overhead=1ull<<20;
        need(encoded.size()<=UINT64_MAX-overhead && manifest.payload_bytes<=UINT64_MAX-encoded.size()-overhead,
             "v3 quota arithmetic overflow");
        const uint64_t reserved=manifest.payload_bytes+encoded.size()+overhead;
        need(quota->held_for(reserved),"v3 quota reservation unavailable");
        r.object=root/rkva_snapshot::hex(p.key.data());
        r.staging=root/("tmp-"+rkva_snapshot::hex(p.transaction.data()));
        need(path_status(r.object).type()==fs::file_type::not_found,"v3 object already exists; refusing overwrite");
        std::error_code ec;
        need(fs::create_directory(r.staging,ec) && !ec,"v3 staging already exists or cannot be created");
        r.state=publication::staging_preserved;
        hold(proof,p,e,store);
        if (!disk->sync_directory(root,why)) throw std::runtime_error("v3 staging directory sync failed: "+why);
        std::array<uint8_t,64*1024> buffer{};
        for(size_t i=0;i<manifest.chunks.size();i++) {
            const auto path=r.staging/("n-"+std::to_string(i)+".kv");
            need(path_status(path).type()==fs::file_type::not_found,"v3 staging chunk already exists");
            auto & c=manifest.chunks[i]; std::ofstream output(path,std::ios::binary|std::ios::trunc);
            need(bool(output),"v3 chunk create failed"); rkva_snapshot::Sha256 hash; uint64_t pos=0;
            while(pos<c.bytes) {
                hold(proof,p,e,store); need(quota->held_for(reserved),"v3 quota lease lost");
                need(input->bound_to(p,e),"v3 source binding lost");
                const auto n=size_t(std::min<uint64_t>(c.bytes-pos,buffer.size()));
                if (!input->read(c,pos,buffer.data(),n,why)) throw std::runtime_error("v3 source read failed: "+why);
                need(bool(output.write(reinterpret_cast<const char *>(buffer.data()),std::streamsize(n))),"v3 chunk write failed");
                hash.update(buffer.data(),n); pos+=n;
            }
            output.flush(); need(bool(output),"v3 chunk flush failed"); output.close(); need(bool(output),"v3 chunk close failed");
            c.digest=hash.finish(); regular(path,c.bytes);
            if (!disk->sync_file(path,why)) throw std::runtime_error("v3 chunk durable sync failed: "+why);
        }
        need(snap::encode(manifest,encoded,&why),"v3 manifest encoding failed");
        const auto path=r.staging/"manifest-v2.bin";
        need(path_status(path).type()==fs::file_type::not_found,"v3 staging manifest already exists");
        std::ofstream output(path,std::ios::binary|std::ios::trunc);
        need(bool(output.write(reinterpret_cast<const char *>(encoded.data()),std::streamsize(encoded.size()))),"v3 manifest write failed");
        output.flush(); need(bool(output),"v3 manifest flush failed"); output.close(); need(bool(output),"v3 manifest close failed");
        if (!disk->sync_file(path,why) || !disk->sync_directory(r.staging,why))
            throw std::runtime_error("v3 staged object sync failed: "+why);
        hold(proof,p,e,store); need(quota->held_for(reserved),"v3 quota lease lost");
        std::string digest; validate_directory(r.staging,p,e,digest);
        hold(proof,p,e,store); need(quota->held_for(reserved),"v3 quota lease lost");
        need(input->bound_to(p,e),"v3 source binding lost");
        need(store->valid_for(root.parent_path()),"concrete store authority lost");
        move_started=true;
        const auto moved=disk->rename_noreplace_and_sync(r.staging,r.object,why);
        if(moved==move_result::no_change) { move_started=false; throw std::runtime_error("v3 publication refused: "+why); }
        if(moved!=move_result::durable) { r.state=publication::publication_uncertain; r.error="v3 publication outcome uncertain: "+why; return r; }
        // An authority loss or verification failure after publication is an
        // uncertain pair, never permission to delete or claim commit.
        hold(proof,p,e,store); need(quota->held_for(reserved),"v3 quota lease lost");
        std::string verify_digest; validate_directory(r.object,p,e,verify_digest);
        need(verify_digest==digest,"v3 published manifest changed");
        need(store->valid_for(root.parent_path()),"concrete store authority lost");
        hold(proof,p,e,store); r.state=publication::published_pending; r.payload_bytes=manifest.payload_bytes;
        r.manifest_sha256=digest; r.staging.clear(); return r;
    } catch(const std::exception & ex) { if(move_started) r.state=publication::publication_uncertain; r.error=ex.what(); return r; }
      catch(...) { if(move_started) r.state=publication::publication_uncertain; r.error="v3 write failed"; return r; }
}
} // namespace rkva_managed_object_v3
