// Read-only forensic inspector for one explicitly identified Qwen35 hybrid
// snapshot profile. It never constructs a llama context or a GPU backend.
#include "native-key-inspect.h"
#include "../../src/llama-kv-mixed-state.h"
#include "../../src/llama-native-prefix-validation.h"
#include "../../ggml/include/rkva-native-state.h"
#include "../../vendor/nlohmann/json.hpp"
#define XXH_INLINE_ALL
#include "../../vendor/hash/xxhash/xxhash.h"
#include <openssl/evp.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#ifdef RKVA_NATIVE_KEY_INSPECT_TESTING
namespace rkva_native_key_inspect { void (*test_mutate)(uint8_t *,uint64_t) = nullptr; }
#endif
namespace {
using json=nlohmann::json;
constexpr const char * profile_model="e2c19a6f1b6193fbab0a2e446b8ee6dd72cd62191ba7158c29c61b40ddbebd5c";
constexpr uint64_t max_file=8ull<<30, max_frame=6ull<<30, max_metadata=65536;
constexpr uint32_t split_p=102400, max_context=162816;
void need(bool v,const char * s) { if(!v) throw std::runtime_error(s); }
uint32_t u32(const uint8_t * p) { return uint32_t(p[0])|uint32_t(p[1])<<8|uint32_t(p[2])<<16|uint32_t(p[3])<<24; }
uint64_t u64(const uint8_t * p) { uint64_t x=0;for(int i=0;i<8;i++)x|=uint64_t(p[i])<<(i*8);return x; }
struct mapping {
    int fd=-1; size_t size=0; const uint8_t * data=nullptr; struct stat initial{};
    explicit mapping(int source_fd) {
        need(source_fd>=0,"invalid input descriptor");
        const int mode=::fcntl(source_fd,F_GETFL);need(mode>=0 && (mode&O_ACCMODE)==O_RDONLY,"input descriptor must be read-only");
        fd=::fcntl(source_fd,F_DUPFD_CLOEXEC,0); need(fd>=0,"cannot duplicate input descriptor");
        try {
            need(::fstat(fd,&initial)==0 && S_ISREG(initial.st_mode) && initial.st_size>0 &&
                 uint64_t(initial.st_size)<=max_file,"input size or type rejected");
            size=size_t(initial.st_size);
            int protection=PROT_READ;
#ifdef RKVA_NATIVE_KEY_INSPECT_TESTING
            if(rkva_native_key_inspect::test_mutate) protection|=PROT_WRITE;
#endif
            void * m=::mmap(nullptr,size,protection,MAP_PRIVATE,fd,0);
            need(m!=MAP_FAILED,"read-only mapping failed");data=static_cast<const uint8_t *>(m);
        } catch (...) { ::close(fd); fd=-1; throw; }
    }
    ~mapping(){if(data) ::munmap(const_cast<uint8_t *>(data),size);if(fd>=0)::close(fd);}
    void unchanged() const {
        struct stat after{};need(::fstat(fd,&after)==0 && initial.st_dev==after.st_dev && initial.st_ino==after.st_ino &&
            initial.st_size==after.st_size && initial.st_mtim.tv_sec==after.st_mtim.tv_sec && initial.st_mtim.tv_nsec==after.st_mtim.tv_nsec &&
            initial.st_ctim.tv_sec==after.st_ctim.tv_sec && initial.st_ctim.tv_nsec==after.st_ctim.tv_nsec,"file changed during verification");
    }
};
struct reader {
    const uint8_t * data; uint64_t limit,pos=0;
    const uint8_t * take(uint64_t n) { need(pos<=limit && n<=limit-pos,"truncated or out-of-bounds frame");const auto * p=data+pos;pos+=n;return p; }
    uint32_t get32(){return u32(take(4));} uint64_t get64(){return u64(take(8));}
};
std::string sha256(const uint8_t * data,size_t size) {
    std::array<uint8_t,32> value{};unsigned n=0;
    need(EVP_Digest(data,size,value.data(),&n,EVP_sha256(),nullptr)==1 && n==32,"SHA-256 failed");
    return rkva_snapshot::hex(value.data());
}
void check_ring(const uint8_t * p,const rkva_native_state::Layout & l) {
    constexpr uint32_t count=rkva_native_state::kOriginalRows;
    std::array<uint32_t,count> pos{};
    for(uint32_t i=0;i<count;i++) { pos[i]=u32(p+4*i);
        if(pos[i]==UINT32_MAX) { need(l.rows<count,"missing native rollback row");continue; }
        need(pos[i]%count==i,"native rollback ring position mismatch"); }
    uint32_t available=l.rows;
    while(available>0 && pos[(available-1)%count]==available-1) --available;
    const uint32_t needed=l.rows>rkva_native_state::kMaxTailTokens?l.rows-rkva_native_state::kMaxTailTokens:0;
    need(available<=needed,"native rollback ring incomplete");
}
void check_sckp(reader & r,uint32_t tokens) {
    if(r.pos==r.limit) return;
    need(r.get32()==0x504b4353 && r.get32()==1,"unexpected or unsupported state appendix");
    const uint32_t count=r.get32();need(count<=1024,"checkpoint count exceeds bound");
    for(uint32_t i=0;i<count;i++) {
        const uint64_t n=r.get64();const int32_t pmin=int32_t(r.get32()),pmax=int32_t(r.get32());
        need(n>0 && n<=tokens && pmin>=0 && pmax>=pmin && uint32_t(pmax)<tokens,"invalid checkpoint metadata");
        bool present=false;
        for(int j=0;j<3;j++){const uint64_t len=r.get64();need(len<=4ull*1024*1024*1024,"checkpoint payload too large");r.take(len);present|=len!=0;}
        need(present,"empty checkpoint");
    }
    need(r.pos==r.limit,"trailing state bytes");
}
rkva_native_key_inspect::result inspect_impl(int fd) {
    auto file=std::make_shared<mapping>(fd);const uint8_t * b=file->data;const uint64_t total=file->size;
#ifdef RKVA_NATIVE_KEY_INSPECT_TESTING
    if(rkva_native_key_inspect::test_mutate) rkva_native_key_inspect::test_mutate(const_cast<uint8_t *>(b),total);
#endif
    need(total>=24,"missing route footer");const uint8_t * footer=b+total-24;
    need(std::memcmp(footer+16,"LLROUTE1",8)==0,"not a unified route envelope");
    const uint64_t mlen=u64(footer);need(mlen>0 && mlen<=max_metadata && mlen<=total-24,"invalid manifest bounds");
    const uint64_t moff=total-24-mlen;need(XXH64(b+moff,size_t(mlen),0)==u64(footer+8),"manifest checksum mismatch");
    const auto m=json::parse(b+moff,b+moff+mlen);
    need(m.at("version").get<uint32_t>()==1 && m.at("model").get<std::string>()==profile_model,"unsupported model identity or envelope version");
    const uint64_t state=m.at("state_bytes").get<uint64_t>();const uint64_t logits=m.value("logits_bytes",uint64_t(0));
    need(state>=12 && state<=moff && logits<=moff-state && state+logits==moff,"invalid payload extents");
    const uint64_t logits_offset=m.value("logits_offset",uint64_t(0));
    const uint64_t logits_checksum=m.value("logits_checksum",uint64_t(0));
    const uint64_t logits_vocab=m.value("logits_vocab",uint64_t(0));
    const uint64_t logits_tokens=m.value("logits_tokens",uint64_t(0));
    if(logits) {
        need(logits_offset==state && logits<=8ull*1024*1024 && logits_vocab>0 &&
            logits_vocab<=UINT32_MAX && logits==4*logits_vocab && logits_tokens==m.at("n_tokens").get<uint64_t>() &&
            XXH64(b+state,size_t(logits),0)==logits_checksum,"invalid logits auxiliary");
    } else need(!logits_offset && !logits_checksum && !logits_vocab && !logits_tokens,"invalid empty logits auxiliary");
    need(XXH64(b,size_t(state),0)==m.at("state_checksum").get<uint64_t>(),"state checksum mismatch");
    const auto layout=json::parse(m.at("layout").get<std::string>());
    const auto & layers=layout.at("mixed_remote_layers");
    need(layout.value("mixed_kv",false) && layout.value("mixed_layer_count",0)==64 &&
         layout.value("mixed_kvarn_type",0)==9 && layout.value("mixed_kvarn_bits_k",0)==4 &&
         layout.value("mixed_kvarn_bits_v",0)==4 && layers.is_array() && layers.size()==16,"unsupported model layout");
    for(uint32_t i=0;i<16;i++)need(layers[i].at("layer").get<uint32_t>()==4*i+3,"unsupported attention layer mapping");
    reader r{b,state};need(r.get32()==0x67677371 && r.get32()==4,"unsupported sequence header");
    const uint32_t tokens=r.get32();need(tokens>split_p && tokens<=max_context &&
         m.at("n_tokens").get<uint64_t>()==tokens && m.at("source_ctx").get<uint32_t>()==max_context &&
         m.at("position").get<int64_t>()==int64_t(tokens)-1,"invalid token prefix metadata");r.take(uint64_t(tokens)*4);
    need(r.get32()==0x4d4b5351 && r.get32()==1,"unsupported mixed envelope");
    const uint64_t frame_size=r.get64();need(frame_size>0 && frame_size<=max_frame,"mixed frame exceeds bound");
    const uint8_t * frame=r.take(frame_size);
    need(frame_size>=104 && u32(frame+28)<=128 && u32(frame+32)==tokens, "mixed frame preliminary bounds rejected");
    llama_kv_mixed::parse_options opts;opts.max_total_bytes=max_frame;opts.max_payload_bytes=max_frame;opts.max_metadata_bytes=64ull<<20;
    auto mixed=llama_kv_mixed::snapshot::parse_view(frame,size_t(frame_size),file,opts);
    need(mixed.version()==2 && mixed.has_split() && mixed.split()->p==split_p && mixed.split()->c==max_context &&
         mixed.layers().size()==32 &&
         mixed.n_used()==tokens && mixed.flags()==llama_kv_mixed::snapshot_flags::full,"mixed split metadata mismatch");
    for (const auto & layer : mixed.layers())
        need(layer.layer_id<64 && layer.layer_id%4==3 && layer.range_id<=1,"unsupported mixed layer set");
    llama_validate_native_prefix(mixed.cells(),tokens,max_context,split_p);
    // The model fingerprint above identifies this exact Qwen35 profile:
    // 48 recurrent R rows then 48 S rows, one cell, F32, no PLE tensors.
    // Other models and recurrent schemas deliberately fail closed.
    need(r.get32()==1 && int32_t(r.get32())==int32_t(tokens)-1 && r.get32()==0 && r.get32()==0 && r.get32()==64,
         "unsupported recurrent framing");
    for(int i=0;i<96;i++) {
        const uint32_t type=r.get32();const uint64_t row=r.get64();
        need(type==0 && row==(i<48?122880ull:3145728ull),"unsupported recurrent row schema");r.take(row);
    }
    const uint64_t native_start=r.pos;const uint8_t * raw=r.take(rkva_native_state::kHeaderBytes);
    rkva_native_state::SetHeader h;std::string error;
    need(rkva_native_state::decode_header(raw,h,&error),"invalid RKVK header");
    need(h.layers==16 && h.layout.origin==split_p && h.layout.capacity==max_context-split_p &&
         h.layout.heads==4 && h.layout.iterations==16 && h.layout.precision_tail_tokens==128 &&
         h.layout.rows==tokens-split_p,"native frame geometry mismatch");
    r.take(h.payload_bytes);const uint8_t * native_footer=r.take(32);
    const uint64_t prefix_end=r.pos;
    need(sha256(raw,size_t(rkva_native_state::kHeaderBytes+h.payload_bytes))==rkva_snapshot::hex(native_footer),"RKVK checksum mismatch");
    for(uint32_t i=0;i<h.layers;i++) {
        const uint64_t positions_offset=native_start+rkva_native_state::kHeaderBytes+uint64_t(i)*h.layer_bytes+h.layer_bytes-rkva_native_state::positions_bytes();
        check_ring(b+positions_offset,h.layout);
    }
    check_sckp(r,tokens);
    const std::string key=sha256(b,size_t(prefix_end));
    const std::string whole=sha256(b,size_t(total));file->unchanged();
    return {tokens,native_start,prefix_end,state,total,key,whole,profile_model,"qwen35-64-recurrent48-f32-v1"};
}
}
namespace rkva_native_key_inspect {
bool inspect(const std::string & path, result & out, std::string & error) {
    if(path.empty() || path.find('\0')!=std::string::npos) {error="invalid path";return false;}
    const int fd=::open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);
    if(fd<0) {error="cannot open regular input";return false;}
    const bool ok=inspect_fd(fd,out,error);::close(fd);return ok;
}
bool inspect_fd(int fd, result & out, std::string & error) {
    try { result candidate=inspect_impl(fd); out=std::move(candidate); error.clear(); return true; }
    catch(const std::exception & e) { error=e.what(); return false; }
}
}
