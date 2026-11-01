#include "native-key-inspect.h"
#include "../../vendor/nlohmann/json.hpp"
#include <iostream>
int main(int argc,char ** argv) {
    if(argc!=2){std::cerr<<"usage: rkva-native-key-inspect SNAPSHOT\n";return 2;}
    rkva_native_key_inspect::result r;std::string error;
    if(!rkva_native_key_inspect::inspect(argv[1],r,error)){std::cerr<<"inspection failed: "<<error<<'\n';return 1;}
    std::cout<<nlohmann::json{{"path",argv[1]},{"tokens",r.tokens},{"native_offset",r.native_offset},
        {"native_end",r.native_end},{"state_bytes",r.state_bytes},{"whole_file_bytes",r.whole_file_bytes},
        {"native_key_sha256",r.native_key_sha256},{"whole_sha256",r.whole_sha256},
        {"model",r.model},{"profile",r.profile}}.dump()<<'\n';
}
