#define RKVA_NATIVE_KEY_INSPECT_TESTING
#include "../tools/rkva-native-key-inspect/native-key-inspect.cpp"
#undef NDEBUG
#include <cassert>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <vector>

static int mutation=0;
static constexpr uint64_t state_bytes=2370799351, recurrent_offset=2011447795, rkva_offset=2168342151;
static void put64(uint8_t * p,uint64_t x){for(int i=0;i<8;i++)p[i]=uint8_t(x>>(8*i));}
static void mutate(uint8_t * p,uint64_t size) {
    assert(size>state_bytes+24);
    const uint64_t len=u64(p+size-24);uint8_t * m=p+size-24-len;
    if(mutation==1){m[2]^=1;return;}
    if(mutation==2){p[100]^=1;return;}
    // Deliberately rebuild both enclosing XXH checksums, so rejection must
    // come from the structured recurrent or RKVK parser, not the envelope.
    const uint64_t offset=mutation==3?recurrent_offset+20:rkva_offset;
    const std::string metadata(reinterpret_cast<const char *>(m),size_t(len));
    const std::string prefix="\"state_checksum\":";
    auto start=metadata.find(prefix);assert(start!=std::string::npos);start+=prefix.size();
    auto end=start;while(end<metadata.size() && metadata[end]>='0' && metadata[end]<='9') ++end;
    assert(end>start);
    bool changed=false;
    const uint8_t old=p[offset];
    for(int v=1;v<256;v++){
        p[offset]=uint8_t(old^v);const auto n=std::to_string(XXH64(p,state_bytes,0));
        if(n.size()==end-start){std::memcpy(m+start,n.data(),n.size());changed=true;break;}
    }
    assert(changed);put64(p+size-16,XXH64(m,size_t(len),0));
}
static void put32(std::vector<uint8_t> & x,uint32_t v){for(int i=0;i<4;i++)x.push_back(uint8_t(v>>(8*i)));}
static void put64v(std::vector<uint8_t> & x,uint64_t v){for(int i=0;i<8;i++)x.push_back(uint8_t(v>>(8*i)));}
int main(int argc,char ** argv) {
    assert(argc==2);
    rkva_native_key_inspect::result out;std::string error;
    assert(rkva_native_key_inspect::inspect(argv[1],out,error));
    assert(out.tokens==110000 && out.native_offset==rkva_offset && out.native_end==state_bytes &&
           out.native_key_sha256=="68b109a4d14bbd26d7937cc9f8268a6bdb706584239531c440762857036d82c5");
    int fd=open(argv[1],O_RDONLY|O_CLOEXEC|O_NOFOLLOW);assert(fd>=0);
    assert(rkva_native_key_inspect::inspect_fd(fd,out,error));assert(fcntl(fd,F_GETFD)>=0);close(fd);
    for(mutation=1;mutation<=4;mutation++){
        rkva_native_key_inspect::test_mutate=mutate;
        rkva_native_key_inspect::result unchanged;unchanged.tokens=77;
        assert(!rkva_native_key_inspect::inspect(argv[1],unchanged,error));
        assert(unchanged.tokens==77 && !error.empty());
        if(mutation==3) assert(error.find("recurrent")!=std::string::npos);
        if(mutation==4) assert(error.find("RKVK")!=std::string::npos);
    }
    rkva_native_key_inspect::test_mutate=nullptr;
    assert(!rkva_native_key_inspect::inspect(std::string("bad\0path",8),out,error));
    // The SCKP parser consumes framed data only and rejects unknown/truncated
    // appendices. These bytes are NOT included in the native prefix hash.
    std::vector<uint8_t> s;put32(s,0x504b4353);put32(s,1);put32(s,1);
    put64v(s,5);put32(s,0);put32(s,4);put64v(s,1);s.push_back(42);put64v(s,0);put64v(s,0);
    reader good{s.data(),s.size()};check_sckp(good,10);assert(good.pos==s.size());
    bool rejected=false;try{reader short_r{s.data(),s.size()-1};check_sckp(short_r,10);}catch(...){rejected=true;}assert(rejected);
    s[0]^=1;rejected=false;try{reader bad{s.data(),s.size()};check_sckp(bad,10);}catch(...){rejected=true;}assert(rejected);
    std::cout<<"native key read-only profile, fd, corruption, framing: PASS\n";
}
