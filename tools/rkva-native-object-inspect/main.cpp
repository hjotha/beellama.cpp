#include "native-object-inspect.h"
#include "../../vendor/nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
using json = nlohmann::json;
namespace {
uint64_t number(const json & object,const char * field,uint64_t max) {
    const auto & value=object.at(field);
    if (!value.is_number_integer()) throw std::runtime_error(std::string("invalid integer: ")+field);
    uint64_t n;
    if (value.is_number_unsigned()) n=value.get<uint64_t>();
    else { const int64_t signed_value=value.get<int64_t>(); if(signed_value<0) throw std::runtime_error(std::string("negative integer: ")+field); n=uint64_t(signed_value); }
    if(n>max) throw std::runtime_error(std::string("integer out of range: ")+field);
    return n;
}
uint32_t u32(const json & object,const char * field) { return uint32_t(number(object,field,UINT32_MAX)); }
rkva_snapshot::Digest key_from_hex(const std::string & text) {
    if(text.size()!=64 || text.find_first_not_of("0123456789abcdef")!=std::string::npos) throw std::runtime_error("key must be 64 lowercase hex digits");
    rkva_snapshot::Digest result{};
    auto nibble=[](char c){ return c<='9' ? c-'0' : c-'a'+10; };
    for(size_t i=0;i<32;i++) result[i]=uint8_t(nibble(text[2*i])*16+nibble(text[2*i+1]));
    return result;
}
} // namespace
int main(int argc,char ** argv) {
    if(argc!=3) { std::cerr << "usage: rkva-native-object-inspect DIRECTORY EXPECTED.json\n"; return 2; }
    try {
        std::ifstream input(argv[2]); if(!input) throw std::runtime_error("cannot open expected identity JSON");
        const auto doc=json::parse(input); const auto & g=doc.at("geometry"); const auto & d=doc.at("descriptor");
        rkva_native_object_inspect::expected expected;
        expected.key=key_from_hex(doc.at("key").get<std::string>());
        expected.live_rows=u32(doc,"live_rows"); expected.saved_generation=number(doc,"saved_generation",UINT64_MAX);
        expected.geometry={u32(g,"n_head"),u32(g,"n_head_kv"),u32(g,"head_dim"),u32(g,"head_slices"),u32(g,"max_ctx"),u32(g,"n_layers")};
        expected.descriptor={u32(d,"origin"),u32(d,"capacity"),u32(d,"storage"),u32(d,"layout_version"),
            u32(d,"token_group"),u32(d,"record_dim"),u32(d,"head_dim"),u32(d,"head_slices"),
            u32(d,"key_bits"),u32(d,"value_bits"),u32(d,"record_bytes"),u32(d,"stage_groups"),
            u32(d,"tail_groups"),u32(d,"sink_tokens"),u32(d,"precision_tail_tokens"),u32(d,"wht_version"),
            u32(d,"domain"),u32(d,"record_policy")};
        rkva_native_object_inspect::result r; std::string error;
        if(!rkva_native_object_inspect::inspect(argv[1],expected,r,error)) {
            std::cerr << "inspection failed: " << error << '\n'; return 1;
        }
        std::cout << json{{"key",r.key},{"chunks",r.chunks},{"payload_bytes",r.payload_bytes},
                          {"manifest_bytes",r.manifest_bytes},{"manifest_sha256",r.manifest_sha256}}.dump() << '\n';
        return 0;
    } catch(const std::exception & e) { std::cerr << "inspection failed: " << e.what() << '\n'; return 1; }
}
