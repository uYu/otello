#include "othello.hpp"
#include "nnue.hpp"
#include "mcts.hpp"
#include "default_engine.hpp"
#include <memory>
#ifdef OTELLO_WITH_EGAROUCID
#include "egaroucid.hpp"
#endif
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <random>
using namespace othello;
namespace {
const char* evaluator(const nnue::Network* network){
    if(network&&network->architecture==nnue::Architecture::Pattern)return network->valueTarget==nnue::ValueTarget::Outcome?"othello-pattern32-outcome-v2":"othello-pattern32-v1.8";
    if(network&&network->cnnPolicy)return "othello-cnn-policy-outcome-v2";
    if(network&&network->trunkBlock)return "othello-nnue-deep-outcome-v5";
    if(network&&network->valueTarget==nnue::ValueTarget::Outcome)return "othello-nnue-outcome-v4";
    if(network&&network->policyHead)return "othello-nnue-policy-v3";
    if(network&&network->architecture==nnue::Architecture::Residual)return "othello-resnet32-v1";
    if(network&&network->architecture==nnue::Architecture::Cnn)return "othello-cnn8x8-v1";
    return !network?"heuristic":network->architecture==nnue::Architecture::Turn?"othello-nnue-v2":"othello-nnue-v1";
}
#ifdef OTELLO_WITH_EGAROUCID
Egaroucid& egaroucid(const std::string& resources){
    static std::unique_ptr<Egaroucid> engine;
    static std::string configured;
    if(!engine){engine=std::make_unique<Egaroucid>(resources);configured=resources;}
    if(configured!=resources)throw std::invalid_argument("Restart protocol to change Egaroucid resources");
    return *engine;
}
#endif
int integer(const std::string& s,int lo,int hi) {
    std::size_t end=0;int n=std::stoi(s,&end);
    if(end!=s.size()||n<lo||n>hi)throw std::invalid_argument("Numeric argument out of range");return n;
}
void jsonString(const std::string& s) {
    std::cout<<'"';for(unsigned char c:s){if(c=='"'||c=='\\')std::cout<<'\\'<<c;else if(c>=32)std::cout<<c;}std::cout<<'"';
}
void state(const Position& p) {
    std::cout<<"\"board\":\""<<board(p)<<"\",\"turn\":"<<p.turn<<",\"black\":"<<count(p.black)<<",\"white\":"<<count(p.white)
        <<",\"hash\":\""<<std::hex<<p.hash<<std::dec<<"\",\"over\":"<<(over(p)?"true":"false")<<",\"mustPass\":"<<(!over(p)&&!legal(p)?"true":"false")<<",\"legalMoves\":[";
    Bits b=legal(p);bool comma=false;while(b){int m=first(b);b&=b-1;if(comma)std::cout<<',';comma=true;std::cout<<'"'<<coordinate(m)<<'"';}std::cout<<']';
}
int request(const std::vector<std::string>& args) {
    try {
        std::string command=args.empty()?"help":args[0];
        Position p;SearchOptions opt;OpeningPolicy opening;
        int move=-2,egLevel=5,bookVariety=1,variety=-1;bool useBook=true,builtinBudget=false,seedSpecified=false;
        std::string engine="builtin",resources,networkFile,searchKind="auto";bool depthSpecified=false;
        for(std::size_t i=1;i<args.size();i+=2) {
            if(i+1==args.size())throw std::invalid_argument("Missing option value");
            auto k=args[i],v=args[i+1];
            if(k=="--board")p=parse(v,p.turn);
            else if(k=="--turn") {if(v=="b"||v=="black"||v=="1")p.turn=1;else if(v=="w"||v=="white"||v=="-1")p.turn=-1;else throw std::invalid_argument("Invalid turn");}
            else if(k=="--move")move=square(v);
            else if(k=="--time"){opt.milliseconds=integer(v,1,60000);builtinBudget=true;}
            else if(k=="--depth"){opt.maxDepth=integer(v,1,64);builtinBudget=true;depthSpecified=true;}
            else if(k=="--search"){if(v!="ab"&&v!="mcts")throw std::invalid_argument("Expected --search ab|mcts");searchKind=v;}
            else if(k=="--level")opt.level=integer(v,0,2);
            else if(k=="--seed"){opt.seed=unsigned(integer(v,0,2147483647));seedSpecified=true;}
            else if(k=="--variety"){if(v!="on"&&v!="off")throw std::invalid_argument("Expected --variety on|off");variety=v=="on";}
            else if(k=="--opening-plies")opening.plies=integer(v,0,60);
            else if(k=="--opening-temperature")opening.temperature=integer(v,0,1000);
            else if(k=="--opening-max-loss")opening.maxLoss=integer(v,0,1000);
            else if(k=="--engine"){if(v!="builtin"&&v!="egaroucid")throw std::invalid_argument("Unknown engine");engine=v;}
            else if(k=="--eg-level")egLevel=integer(v,1,60);
            else if(k=="--book-variety")bookVariety=integer(v,0,3);
            else if(k=="--resources")resources=v;
            else if(k=="--network")networkFile=v;
            else if(k=="--book"){if(v!="on"&&v!="off")throw std::invalid_argument("Expected --book on|off");useBook=v=="on";}
            else throw std::invalid_argument("Unknown option: "+k);
        }
        p.hash=hashBoard(p.black,p.white,p.turn);
        if(searchKind=="mcts"&&(engine!="builtin"||depthSpecified))throw std::invalid_argument("MCTS uses builtin --time; --depth and Egaroucid are unsupported");
        if(engine=="egaroucid"&&builtinBudget)throw std::invalid_argument("Egaroucid does not implement time_limit_ms; use --eg-level instead of --time/--depth");
        if(engine=="egaroucid"&&!networkFile.empty())throw std::invalid_argument("--network is for the builtin Othello evaluator; Egaroucid owns its evaluation model");
        std::unique_ptr<nnue::Network> customNetwork;
        const nnue::Network* network=nullptr;
        if(!networkFile.empty()&&networkFile!="heuristic"){
            customNetwork=std::make_unique<nnue::Network>(nnue::Network::load(networkFile));network=customNetwork.get();
        }else if(networkFile.empty()&&engine=="builtin"&&opt.level==2&&(command=="analyze"||command=="evaluate"||command=="selfplay"))network=&defaultNetwork();
        opt.network=network;
        if(searchKind=="auto")searchKind=network&&!depthSpecified?"mcts":"ab";
        bool useVariety=variety<0?command=="selfplay"&&bool(network):variety==1;
        if(useVariety&&!network)throw std::invalid_argument("Opening variety requires an NNUE network");
        if(useVariety&&!seedSpecified)opt.seed=std::random_device{}()&0x7fffffffU;
        auto runSearch=[&](const Position& position){return searchKind=="mcts"?mcts(position,{opt.milliseconds,0,opt.seed,opt.network}).analysis:search(position,opt);};
#ifndef OTELLO_WITH_EGAROUCID
        if(engine=="egaroucid")throw std::runtime_error("Egaroucid is not built; run sh scripts/setup-egaroucid.sh");
#endif
        if(command=="help"||command=="--help") {
            std::cout<<"{\"name\":\"otello\",\"version\":1,\"commands\":[\"legal\",\"play\",\"analyze\",\"evaluate\",\"selfplay\",\"protocol\"],\"coordinates\":\"A1-H8, A1 = first cell, B/W/. row-major\",\"options\":\"--board BOARD --turn b|w --move D3|pass --time 800 --depth 64 --level 0|1|2 --seed 1 --search ab|mcts --engine builtin|egaroucid --eg-level 5 --book on|off --book-variety 0|1|2|3 --resources PATH --network MODEL --variety on|off --opening-plies 12 --opening-temperature 32 --opening-max-loss 64\"}\n";return 0;
        }
        if(command=="evaluate"){
            std::cout<<"{\"score\":"<<(network?network->evaluate(p):evaluate(p))<<",\"evaluator\":\""<<evaluator(network)<<"\",\"scoreUnit\":\"evaluation\",";state(p);std::cout<<"}\n";
        }
        else if(command=="legal") {std::cout<<'{';state(p);std::cout<<"}\n";}
        else if(command=="play") {
            if(move==-2)throw std::invalid_argument("play requires --move");
            auto next=play(p,move);std::cout<<'{';state(next);std::cout<<",\"flipped\":"<<count(flips(p,move))<<"}\n";
        } else if(command=="analyze") {
#ifdef OTELLO_WITH_EGAROUCID
            if(engine=="egaroucid"){
                auto& eg=egaroucid(resources);auto r=eg.analyze(p,egLevel,useBook,bookVariety);
                std::cout<<"{\"engine\":\"egaroucid\",\"engineVersion\":";jsonString(eg.version());
                std::cout<<",\"move\":\""<<coordinate(r.move)<<"\",\"moveIndex\":"<<r.move<<",\"score\":"<<r.value<<",\"scoreUnit\":\"discs\",\"depth\":"<<r.depth
                    <<",\"nodes\":"<<r.nodes<<",\"ttHits\":null,\"elapsedMs\":"<<r.elapsed<<",\"exact\":"<<(r.exact?"true":"false")
                    <<",\"useBook\":"<<(useBook?"true":"false")<<",\"bookVariety\":"<<bookVariety<<",\"bookHit\":"<<(r.bookHit?"true":"false")<<",\"candidates\":[],";
                state(p);std::cout<<"}\n";return 0;
            }
#endif
            auto r=runSearch(p);auto selected=selectMove(p,r,useVariety?opening:OpeningPolicy{0,32,64},opt.seed);
            r.move=selected.move;r.score=selected.score;
            std::cout<<"{\"search\":\""<<searchKind<<"\",\"depthKind\":\""<<(searchKind=="mcts"?"maximumVisited":"completed")<<"\",\"seed\":"<<opt.seed<<",\"selection\":{\"eligible\":"<<selected.eligible<<",\"temperatureMilli\":"<<selected.temperatureMilli
                <<",\"lossLimit\":"<<selected.lossLimit<<",\"bestScore\":"<<selected.bestScore<<",\"scoreLoss\":"<<selected.bestScore-selected.score
                <<",\"sampled\":"<<(selected.sampled?"true":"false")<<"},\"move\":\""<<coordinate(r.move)<<"\",\"score\":"<<r.score<<",\"depth\":"<<r.depth<<",\"nodes\":"<<r.nodes
                <<",\"ttHits\":"<<r.ttHits<<",\"elapsedMs\":"<<r.elapsed<<",\"exact\":"<<(r.exact?"true":"false")<<",\"candidates\":[";
            for(std::size_t i=0;i<r.candidates.size();++i){if(i)std::cout<<',';std::cout<<"{\"move\":\""<<coordinate(r.candidates[i].move)<<"\",\"score\":"<<r.candidates[i].score<<'}';}
            std::cout<<"],\"evaluator\":\""<<evaluator(network)<<"\",\"searchStats\":{\"ttCutoffs\":"<<r.ttCutoffs<<",\"betaCutoffs\":"<<r.betaCutoffs<<",\"pvsResearches\":"<<r.pvsResearches<<",\"aspirationRetries\":"<<r.aspirationRetries<<",\"evaluations\":"<<r.evaluations<<"},";state(p);std::cout<<"}\n";
        } else if(command=="selfplay") {
            std::vector<int> moves;
            while(!over(p)) {
                int m=-1;
                if(legal(p)){
#ifdef OTELLO_WITH_EGAROUCID
                    if(engine=="egaroucid")m=egaroucid(resources).analyze(p,egLevel,useBook,bookVariety).move;
                    else
#endif
                    {auto r=runSearch(p);m=selectMove(p,r,useVariety?opening:OpeningPolicy{0,32,64},opt.seed).move;}
                }
                moves.push_back(m);p=play(p,m);
            }
            std::cout<<"{\"search\":\""<<searchKind<<"\",\"evaluator\":\""<<evaluator(network)<<"\",\"seed\":"<<opt.seed<<",\"variety\":"<<(useVariety?"true":"false")<<',';state(p);std::cout<<",\"moves\":[";
            for(std::size_t i=0;i<moves.size();++i){if(i)std::cout<<',';std::cout<<'"'<<coordinate(moves[i])<<'"';}std::cout<<"]}\n";
        } else throw std::invalid_argument("Unknown command");
        return 0;
    } catch(const std::exception& e) {std::cout<<"{\"error\":";jsonString(e.what());std::cout<<"}\n";return 1;}
}
}
int main(int argc,char** argv) {
    std::vector<std::string> args(argv+1,argv+argc);
    if(!args.empty()&&args[0]=="protocol") {
        std::string line;
        while(std::getline(std::cin,line)) {
            if(line=="quit")break;
            std::istringstream input(line);std::string token;std::vector<std::string> words;
            while(input>>token)words.push_back(token);
            if(!words.empty())request(words);
            std::cout.flush();
        }
        return 0;
    }
    return request(args);
}
