#include "othello.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <filesystem>
#include <cctype>
using namespace othello;
namespace {
void quote(std::ostream& out,const std::string& text){out<<'"';for(unsigned char c:text){if(c=='"'||c=='\\')out<<'\\'<<c;else if(c>=32)out<<c;}out<<'"';}
std::uint64_t hash(const std::string& text){std::uint64_t value=14695981039346656037ULL;for(unsigned char c:text){value^=c;value*=1099511628211ULL;}return value;}
int transform(int move,int symmetry){if(move<0)return 64;int x=move%8,y=move/8;if(symmetry&4)x=7-x;for(int r=0;r<(symmetry&3);++r){int old=x;x=7-y;y=old;}return y*8+x;}
std::string canonical(const std::vector<int>& moves,int limit=1000){
    std::string best;for(int symmetry=0;symmetry<8;++symmetry){std::string value;int n=0;
        for(int m:moves){value+=char(transform(m,symmetry));if(m>=0&&++n>=limit)break;}
        if(symmetry==0||value<best)best=value;
    }return best;
}
struct Writer {
    std::ostream& out;std::unordered_set<std::string> seen;
    std::uint64_t records=0,accepted=0,rejected=0,duplicates=0,incomplete=0,scoreMismatch=0,positions=0;
    std::uint64_t maxGames=0;
    bool done()const{return maxGames&&accepted>=maxGames;}
    void game(const std::vector<std::pair<int,int>>& transcript,int result,int theoretical,const std::string& source,const std::string& file){
        ++records;
        try{
            if(result < -64||result > 64||transcript.empty())throw std::runtime_error("Invalid result or empty game");
            Position p;std::vector<int> actions;
            for(auto [move,actor]:transcript){
                if(over(p))throw std::runtime_error("Moves after terminal position");
                if(!legal(p)){actions.push_back(-1);p=play(p,-1);}
                if(actor&&actor!=p.turn)throw std::runtime_error("Wrong player sign");
                p=play(p,move);actions.push_back(move);
            }
            auto key=canonical(actions);
            if(over(p)&&count(p.black|p.white)==64&&count(p.black)-count(p.white)!=result){++scoreMismatch;return;}
            if(!seen.insert(key).second){++duplicates;return;}
            if(!over(p))++incomplete;
            ++accepted;positions+=transcript.size();
            auto splitHash=hash(canonical(actions,8));int bucket=int(splitHash%100);
            const char* split=bucket<80?"train":bucket<90?"validation":"test";
            out<<"{\"version\":1,\"id\":\""<<std::hex<<hash(key)<<std::dec<<"\",\"split\":\""<<split<<"\",\"source\":";quote(out,source);
            out<<",\"sourceFile\":";quote(out,std::filesystem::path(file).filename().string());
            out<<",\"blackResult\":"<<result<<",\"theoreticalBlackResult\":";
            if(theoretical>=-64&&theoretical<=64)out<<theoretical;else out<<"null";
            out<<",\"complete\":"<<(over(p)?"true":"false")<<",\"moves\":[";
            for(std::size_t i=0;i<actions.size();++i){if(i)out<<',';out<<actions[i];}out<<"]}\n";
        }catch(const std::exception&){++rejected;}
    }
};
std::uint32_t little(const unsigned char* bytes){return std::uint32_t(bytes[0])|std::uint32_t(bytes[1])<<8|std::uint32_t(bytes[2])<<16|std::uint32_t(bytes[3])<<24;}
void wthor(const std::string& path,Writer& writer){
    std::ifstream in(path,std::ios::binary);unsigned char header[16]{};in.read(reinterpret_cast<char*>(header),16);
    if(!in||(header[12]!=8&&header[12]!=0)||header[13]!=0)throw std::runtime_error("Expected an 8x8 WTHOR game archive");
    auto count=little(header+4);
    if(std::filesystem::file_size(path)!=16+std::uint64_t(count)*68)throw std::runtime_error("WTHOR file length does not match header");
    for(std::uint32_t n=0;n<count&&!writer.done();++n){
        unsigned char record[68]{};in.read(reinterpret_cast<char*>(record),68);std::vector<std::pair<int,int>> moves;bool padding=false,invalid=false;
        for(int i=8;i<68;++i){int value=record[i];if(!value){padding=true;continue;}
            int y=value/10-1,x=value%10-1;
            if(padding||x<0||x>7||y<0||y>7){invalid=true;break;}moves.push_back({y*8+x,0});}
        if(invalid||record[6]>64){++writer.records;++writer.rejected;continue;}
        writer.game(moves,2*int(record[6])-64,record[7]<=64?2*int(record[7])-64:1000,"wthor",path);
    }
}
void logistello(const std::string& path,Writer& writer){
    std::ifstream in(path);if(!in)throw std::runtime_error("Cannot read Logistello transcript file");std::string line;
    while(!writer.done()&&std::getline(in,line)){
        if(line.empty())continue;
        auto colon=line.find(':');std::vector<std::pair<int,int>> moves;
        try{
            if(colon==std::string::npos)throw std::runtime_error("Missing result");
            std::string sequence=line.substr(0,colon);while(!sequence.empty()&&std::isspace(static_cast<unsigned char>(sequence.back())))sequence.pop_back();
            if(sequence.size()%3)throw std::runtime_error("Invalid move sequence");
            for(std::size_t i=0;i<sequence.size();i+=3){if(sequence[i]!='+'&&sequence[i]!='-')throw std::runtime_error("Missing player sign");moves.push_back({square(sequence.substr(i+1,2)),sequence[i]=='+'?1:-1});}
            std::istringstream result(line.substr(colon+1));int value;if(!(result>>value))throw std::runtime_error("Invalid result");
            writer.game(moves,value,1000,"logistello",path);
        }catch(const std::exception&){++writer.records;++writer.rejected;}
    }
}
}
int main(int argc,char** argv){
    try{
        std::vector<std::string> inputs;std::string output;std::uint64_t maxGames=0;
        for(int i=1;i<argc;i+=2){if(i+1==argc)throw std::runtime_error("Missing option value");std::string key=argv[i];
            if(key=="--input")inputs.emplace_back(argv[i+1]);else if(key=="--output")output=argv[i+1];
            else if(key=="--max-games"){std::string value=argv[i+1];std::size_t end=0;if(value.empty()||value[0]=='-')throw std::runtime_error("Invalid max-games");maxGames=std::stoull(value,&end);if(end!=value.size())throw std::runtime_error("Invalid max-games");}
            else throw std::runtime_error("Unknown option");}
        if(inputs.empty()||output.empty())throw std::runtime_error("Usage: otello_dataset --input FILE.wtb --input FILE.gam --output games.jsonl [--max-games N]");
        for(const auto& file:inputs)if(std::filesystem::absolute(file).lexically_normal()==std::filesystem::absolute(output).lexically_normal())throw std::runtime_error("Output cannot overwrite input");
        std::ofstream out(output);if(!out)throw std::runtime_error("Cannot create output");Writer writer{out};writer.maxGames=maxGames;
        for(const auto& file:inputs){if(writer.done())break;auto ext=std::filesystem::path(file).extension().string();
            if(ext==".wtb"||ext==".WTB")wthor(file,writer);else if(ext==".gam")logistello(file,writer);else throw std::runtime_error("Expected uncompressed .wtb or .gam");}
        out.flush();if(!out)throw std::runtime_error("Cannot write dataset output");
        std::cout<<"{\"records\":"<<writer.records<<",\"accepted\":"<<writer.accepted<<",\"rejected\":"<<writer.rejected<<",\"duplicates\":"<<writer.duplicates<<",\"incomplete\":"<<writer.incomplete<<",\"scoreMismatch\":"<<writer.scoreMismatch<<",\"placements\":"<<writer.positions<<"}\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
