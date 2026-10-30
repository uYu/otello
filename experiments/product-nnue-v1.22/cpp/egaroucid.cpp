#include "egaroucid.hpp"
#include <egaroucid/egaroucid.h>
#include <climits>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
extern "C" int otello_egaroucid_book_probe(const int*,int,int,unsigned,int*,int*);

namespace {
std::mutex engineMutex;
std::string initializedResources;
thread_local std::string lastError;
std::array<int,64> encode(const othello::Position& p) {
    std::array<int,64> cells{};
    for(int i=0;i<64;++i)cells[i]=(p.black>>i&1)?EGAROUCID_BLACK:(p.white>>i&1)?EGAROUCID_WHITE:EGAROUCID_EMPTY;
    return cells;
}
void validateResources(const std::filesystem::path& path) {
    for(const char* file:{"eval.egev2","eval_move_ordering_end.egev","book.egbk3","hash/hash25.eghs"})
        if(!std::filesystem::is_regular_file(path/file))throw std::runtime_error(std::string("Missing Egaroucid resource: ")+file);
    std::ifstream input(path/"book.egbk3",std::ios::binary);
    unsigned char header[14]{};input.read(reinterpret_cast<char*>(header),14);
    std::uint32_t n=std::uint32_t(header[10])|(std::uint32_t(header[11])<<8)|(std::uint32_t(header[12])<<16)|(std::uint32_t(header[13])<<24);
    if(!input||std::string(reinterpret_cast<char*>(header),9)!="DICUORAGE"||header[9]!=3||n==0||std::filesystem::file_size(path/"book.egbk3")!=14+std::uint64_t(n)*25)
        throw std::runtime_error("Invalid or truncated Egaroucid opening book");
}
egaroucid_search_result run(egaroucid_engine* engine,const othello::Position& p,int level,bool useBook,int bookVariety=0) {
    auto cells=encode(p);
    egaroucid_search_options opt{};opt.size=sizeof(opt);opt.level=level;
    opt.use_book=useBook?1:0;opt.book_accuracy_level=bookVariety;opt.use_multi_thread=0;opt.show_log=0;
    opt.time_limit_ms=-1; // Upstream accepts but does not implement this field.
    egaroucid_search_result out{};out.size=sizeof(out);
    auto status=egaroucid_search_array(engine,cells.data(),p.turn==1?EGAROUCID_BLACK:EGAROUCID_WHITE,&opt,&out);
    if(status!=EGAROUCID_OK)throw std::runtime_error("Egaroucid search failed: "+std::to_string(status));
    auto moves=othello::legal(p);
    if(out.move==-1?moves!=0:out.move<0||out.move>63||!(moves&(1ULL<<out.move)))
        throw std::runtime_error("Egaroucid returned an illegal move");
    return out;
}
}
namespace othello {
struct Egaroucid::Impl {
    egaroucid_engine* engine=nullptr;
    ~Impl(){if(engine)egaroucid_destroy(engine);}
};
std::string Egaroucid::defaultResources() {
    const char* configured=std::getenv("OTELLO_EGAROUCID_RESOURCES");
    return configured&&*configured?configured:OTELLO_EGAROUCID_RESOURCES;
}
Egaroucid::Egaroucid(const std::string& resources):impl(std::make_unique<Impl>()) {
    auto path=std::filesystem::canonical(resources.empty()?defaultResources():resources);
    std::lock_guard<std::mutex> guard(engineMutex);
    validateResources(path);
    if(initializedResources.empty()) {
        if(egaroucid_global_init(path.string().c_str())!=EGAROUCID_OK)throw std::runtime_error("Egaroucid resource initialization failed");
        impl->engine=egaroucid_create();
        if(!impl->engine)throw std::runtime_error("Egaroucid engine allocation failed");
        auto probe=run(impl->engine,Position{},1,true);
        if(probe.depth!=-1)throw std::runtime_error("Egaroucid opening book did not produce an opening hit");
        initializedResources=path.string();
    } else {
        if(initializedResources!=path.string())throw std::runtime_error("Egaroucid resources cannot change inside a running process");
        impl->engine=egaroucid_create();
        if(!impl->engine)throw std::runtime_error("Egaroucid engine allocation failed");
    }
}
Egaroucid::~Egaroucid()=default;
std::string Egaroucid::version() const{return egaroucid_version();}
bool Egaroucid::probeBook(const Position& p,EgaroucidResult& result,int variety,unsigned seed){
    if(variety<0||variety>3||(p.turn!=1&&p.turn!=-1)||(p.black&p.white))throw std::invalid_argument("Invalid book lookup arguments");
    auto start=std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> guard(engineMutex);
    result=EgaroucidResult{};auto cells=encode(p);
    int found=otello_egaroucid_book_probe(cells.data(),p.turn==1?EGAROUCID_BLACK:EGAROUCID_WHITE,variety,seed,&result.move,&result.value);
    if(found<0)throw std::runtime_error("Egaroucid book lookup failed: "+std::to_string(found));
    if(found&&(result.move<0||result.move>63||!(legal(p)&(1ULL<<result.move))))throw std::runtime_error("Book returned an illegal move");
    result.bookHit=found==1;result.depth=found?-1:0;
    result.elapsed=int(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count());
    return result.bookHit;
}
EgaroucidResult Egaroucid::analyze(const Position& p,int level,bool useBook,int bookVariety) {
    if(level<1||level>60)throw std::invalid_argument("Egaroucid level must be 1..60");
    if(bookVariety<0||bookVariety>3)throw std::invalid_argument("Book variety must be 0..3");
    if(p.black&p.white)throw std::invalid_argument("Overlapping discs");
    if(p.turn!=1&&p.turn!=-1)throw std::invalid_argument("Invalid turn");
    auto start=std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> guard(engineMutex);
    auto r=run(impl->engine,p,level,useBook,bookVariety);
    return {r.move,r.value,r.depth,int(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count()),r.nodes,r.depth==-1,r.is_end_search!=0};
}
}
extern "C" {
void* ot_egaroucid_create(const char* resources) {
    try{auto* p=new othello::Egaroucid(resources?resources:"");lastError.clear();return p;}
    catch(const std::exception& e){lastError=e.what();return nullptr;}
}
void ot_egaroucid_destroy(void* engine){delete static_cast<othello::Egaroucid*>(engine);}
int ot_egaroucid_book(void* engine,const int* cells,int turn,int variety,unsigned seed,int* output){
    try{
        if(!engine||!cells||!output||(turn!=1&&turn!=-1))throw std::invalid_argument("Invalid book arguments");
        std::string board(64,'.');
        for(int i=0;i<64;++i){if(cells[i]!=0&&cells[i]!=1&&cells[i]!=-1)throw std::invalid_argument("Invalid cell");board[i]=cells[i]==1?'B':cells[i]==-1?'W':'.';}
        othello::EgaroucidResult r;
        bool hit=static_cast<othello::Egaroucid*>(engine)->probeBook(othello::parse(board,turn),r,variety,seed);
        output[0]=r.move;output[1]=r.value;output[2]=r.depth;output[3]=r.elapsed;output[4]=0;output[5]=hit;output[6]=0;output[7]=1;
        lastError.clear();return hit?1:0;
    }catch(const std::exception& e){lastError=e.what();return -1;}
}
int ot_egaroucid_search(void* engine,const int* cells,int turn,int level,int useBook,int* output) {
    try{
        if(!engine||!cells||!output||(turn!=1&&turn!=-1)||(useBook!=0&&useBook!=1))throw std::invalid_argument("Invalid Egaroucid arguments");
        std::string board(64,'.');
        for(int i=0;i<64;++i){if(cells[i]!=0&&cells[i]!=1&&cells[i]!=-1)throw std::invalid_argument("Invalid cell");board[i]=cells[i]==1?'B':cells[i]==-1?'W':'.';}
        auto r=static_cast<othello::Egaroucid*>(engine)->analyze(othello::parse(board,turn),level,useBook);
        output[0]=r.move;output[1]=r.value;output[2]=r.depth;output[3]=r.elapsed;output[4]=int(std::min<std::uint64_t>(INT_MAX,r.nodes));output[5]=r.bookHit;output[6]=r.exact;output[7]=useBook;
        lastError.clear();return 1;
    }catch(const std::exception& e){lastError=e.what();return -1;}
}
const char* ot_egaroucid_last_error(){return lastError.c_str();}
}
