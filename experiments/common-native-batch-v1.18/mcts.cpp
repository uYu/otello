#include "mcts.hpp"
#include "nnue.hpp"
#include <cmath>
#include <limits>
#include <memory>
#include <functional>
#include "native_batch.hpp"

// Experimental bridge only. A registered callback evaluates a complete batch
// synchronously; the original model/search ABI remains available unchanged.
using BatchCallback=int(*)(int,const int*,const int*,const std::uint64_t*,int*,double*);
namespace {thread_local BatchCallback batchCallback=nullptr;thread_local int batchSize=1;
    thread_local unsigned batchCalls=0,batchLeaves=0;}
extern "C" int ot_cnn_set_batch_evaluator(BatchCallback callback,int size){
    if(size<1||size>128)return -1;
    batchCallback=callback;batchSize=size;batchCalls=batchLeaves=0;return 1;
}
extern "C" int ot_cnn_set_native_batch_evaluator(int size){return ot_cnn_set_batch_evaluator(nativeCnnBatch,size);}
extern "C" void ot_cnn_batch_stats(unsigned* calls,unsigned* leaves){if(calls)*calls=batchCalls;if(leaves)*leaves=batchLeaves;}

namespace othello {
namespace {
struct Node {
    Position position;
    std::uint64_t visits=0;
    double total=0,value=0;
    int firstChild=-1,children=0,move=-1;
    bool solved=false,pending=false;
};
struct Edge {int child=-1,move=-1;double prior=0;};
bool samePosition(const Position& a,const Position& b){return a.hash==b.hash&&a.black==b.black&&a.white==b.white&&a.turn==b.turn;}
struct Cache {
    std::vector<Node> tree;
    std::vector<Edge> edges;
    std::unique_ptr<nnue::Network> evaluator;
    void reset(const Position& p){tree.clear();edges.clear();Node n;n.position=p;tree.push_back(n);}
    std::size_t prepare(const Position& p,const nnue::Network* network){
        const bool same=network&&evaluator?network->sameParameters(*evaluator):!network&&!evaluator;
        if(!same){tree.clear();edges.clear();evaluator=network?network->snapshot():nullptr;}
        if(!tree.empty()&&samePosition(tree[0].position,p))return tree.size();
        int root=-1;
        for(std::size_t i=0;i<tree.size();++i)if(samePosition(tree[i].position,p)){root=int(i);break;}
        if(root<0){reset(p);return 0;}
        std::vector<Node> kept;std::vector<Edge> keptEdges;
        kept.reserve(tree.size());keptEdges.reserve(edges.size());kept.push_back(tree[root]);
        for(std::size_t i=0;i<kept.size();++i){
            int old=kept[i].firstChild,n=kept[i].children;
            if(old>=0){kept[i].firstChild=int(keptEdges.size());
                for(int j=0;j<n;++j){Edge edge=edges[old+j];
                    if(edge.child>=0){int child=edge.child;edge.child=int(kept.size());kept.push_back(tree[child]);}
                    keptEdges.push_back(edge);
                }
            }
        }
        tree.swap(kept);edges.swap(keptEdges);return tree.size();
    }
};
Cache& cache(){thread_local Cache instance;return instance;}
double mean(const Node& n){return n.solved?n.value:n.visits?n.total/n.visits:0;}
int score(double value,bool exact,bool outcome){
    if(!exact)return int(std::lround(value*1000));
    if(outcome)return value>0?100000:value<0?-100000:0;
    int discs=int(std::lround(value*64));return discs>0?100000+discs*100:discs<0?-100000+discs*100:0;
}
std::uint64_t tie(const Position& p,int move,std::uint32_t seed){Bits x=p.hash^(Bits(seed)<<32)^Bits(move+1);return splitmix(x);}
void solve(int index,Cache& shared,bool outcome){
    auto& n=shared.tree[index];if(n.firstChild<0||!n.children)return;
    double best=-2;
    if(outcome){
        bool allSolved=true;
        for(int j=0;j<n.children;++j){int child=shared.edges[n.firstChild+j].child;
            if(child<0||!shared.tree[child].solved){allSolved=false;continue;}
            best=std::max(best,-shared.tree[child].value);
            // One proven winning move proves a win; margin is not an objective.
            if(best==1){n.solved=true;n.value=1;return;}
        }
        if(allSolved){n.solved=true;n.value=best;}return;
    }
    for(int j=0;j<n.children;++j){int child=shared.edges[n.firstChild+j].child;
        if(child<0||!shared.tree[child].solved)return;best=std::max(best,-shared.tree[child].value);}
    n.solved=true;n.value=best;
}
}
void clearMctsCache(){auto& c=cache();c.tree.clear();c.edges.clear();c.evaluator.reset();}
MctsResult mcts(const Position& position,MctsOptions options){
    const bool outcome=options.network&&options.network->valueTarget==nnue::ValueTarget::Outcome;
    const bool batching=batchCallback&&options.network&&options.network->hasPolicy();
    activeBatchCnn=options.network?options.network->cnnPolicy.get():nullptr;
    activeBatchNetwork=options.network;
    if(options.milliseconds<0||options.milliseconds>60000||(!options.milliseconds&&!options.simulations)
       ||!std::isfinite(options.exploration)||options.exploration<=0||options.maxNodes<65||options.maxNodes>2097152)
        throw std::invalid_argument("Invalid MCTS budget/exploration/node limit");
    const auto start=std::chrono::steady_clock::now();const auto deadline=start+std::chrono::milliseconds(options.milliseconds);
    MctsResult result;auto& shared=cache();result.reusedNodes=shared.prepare(position,options.network);
    if(batching&&batchCallback==nativeCnnBatch)prepareNativeModel(*options.network);
    auto& tree=shared.tree;auto& edges=shared.edges;
    if(tree.size()>options.maxNodes){shared.reset(position);result.reusedNodes=0;}
    auto& analysis=result.analysis;double totalDepth=0;
    struct Pending {std::array<int,130> path;int length;Position position;Bits moves;bool reserved=true;};
    std::vector<Pending> pending;
    auto rollback=[&](){
        for(auto& job:pending){if(!job.reserved)continue;tree[job.path[job.length-1]].pending=false;
            for(int i=0;i<job.length;++i)--tree[job.path[i]].visits;job.reserved=false;}
    };
    auto backup=[&](const std::array<int,130>& path,int length,double value){
        for(int i=length-1;i>=0;--i){auto& n=tree[path[i]];n.total+=value;++n.visits;solve(path[i],shared,outcome);value=-value;}
        ++analysis.nodes;analysis.depth=std::max(analysis.depth,length-1);totalDepth+=length-1;
    };
    auto expand=[&](int current,const Position& p,Bits moves,const std::array<double,65>& priors){
        tree[current].firstChild=int(edges.size());tree[current].children=moves?count(moves):1;
        if(!moves)edges.push_back({-1,-1,1});
        else while(moves){int move=first(moves);moves&=moves-1;edges.push_back({-1,move,priors[move]});}
    };
    auto flush=[&](){
        if(pending.empty())return;
        const int size=int(pending.size());std::vector<int> boards(size*64),turns(size),scores(size);
        std::vector<Bits> masks(size);std::vector<double> probabilities(size*65);
        for(int k=0;k<size;++k){const auto& p=pending[k].position;turns[k]=p.turn;masks[k]=pending[k].moves;
            for(int j=0;j<64;++j)boards[k*64+j]=int((p.black>>j)&1)-int((p.white>>j)&1);}
        if(batchCallback(size,boards.data(),turns.data(),masks.data(),scores.data(),probabilities.data())!=1)
            throw std::runtime_error("Batch CNN evaluator failed");
        ++batchCalls;batchLeaves+=size;
        for(int k=0;k<size;++k){double sum=0;
            if(scores[k]<-1000||scores[k]>1000)throw std::runtime_error("Invalid batch value");
            for(int m=0;m<65;++m){double p=probabilities[k*65+m];
                bool legal=pending[k].moves?(m<64&&(pending[k].moves&(1ULL<<m))):m==64;
                if(!std::isfinite(p)||p<0||p>1||(!legal&&p!=0))throw std::runtime_error("Invalid batch policy");sum+=p;}
            if(std::abs(sum-1)>1e-5)throw std::runtime_error("Batch policy not normalized");}
        rollback();
        for(int k=0;k<size;++k){const auto& job=pending[k];int current=job.path[job.length-1];
            std::array<double,65> priors{};std::copy_n(probabilities.data()+k*65,65,priors.data());
            expand(current,job.position,job.moves,priors);++analysis.evaluations;
            backup(job.path,job.length,double(scores[k])/1000);}
        pending.clear();
    };
    try{
    while(!tree[0].solved&&(!options.simulations||analysis.nodes<options.simulations)){
        if(analysis.nodes&&options.milliseconds&&std::chrono::steady_clock::now()>=deadline)break;
        std::array<int,130> path{};int length=1,current=0;path[0]=0;
        while(tree[current].firstChild>=0&&!tree[current].solved){
            const auto& parent=tree[current];int chosen=-1;double best=-std::numeric_limits<double>::infinity();
            const bool learned=options.network&&options.network->hasPolicy();
            const double exploration=options.exploration*std::sqrt(double(std::max<std::uint64_t>(1,parent.visits)));
            for(int j=0;j<parent.children;++j){
                int index=parent.firstChild+j;const Edge& edge=edges[index];
                std::uint64_t visits=edge.child<0?0:tree[edge.child].visits;
                if(batching&&edge.child>=0&&tree[edge.child].pending)continue;
                const double prior=learned?exploration*edge.prior:exploration/parent.children;
                double utility=visits?-mean(tree[edge.child])+prior/(1+visits):std::numeric_limits<double>::infinity();
                bool preferred=chosen>=0&&learned&&utility==best&&edge.prior!=edges[chosen].prior;
                if(chosen<0||utility>best||(utility==best&&(preferred?edge.prior>edges[chosen].prior:
                    tie(parent.position,edge.move,options.seed)>tie(parent.position,edges[chosen].move,options.seed)))){best=utility;chosen=index;}
            }
            if(chosen<0)break; // All immediate children are awaiting this batch.
            if(length>=int(path.size()))throw std::runtime_error("MCTS path exceeds game length");
            if(edges[chosen].child<0){
                if(tree.size()>=options.maxNodes){result.memoryLimited=true;break;}
                Node child;child.move=edges[chosen].move;child.position=play(tree[current].position,child.move);
                edges[chosen].child=int(tree.size());tree.push_back(child);
            }
            current=edges[chosen].child;path[length++]=current;
        }
        if(result.memoryLimited)break;
        if(tree[current].pending||(!tree[current].solved&&tree[current].firstChild>=0)){
            if(pending.empty())throw std::runtime_error("Blocked batch without pending jobs");
            flush();continue;
        }
        double value;
        if(tree[current].solved)value=tree[current].value;
        else{
            const Position p=tree[current].position;Bits moves=legal(p);bool terminal=!moves&&!legal(p.opponent(),p.own());
            if(terminal){int difference=count(p.own())-count(p.opponent());
                value=outcome?double((difference>0)-(difference<0)):double(difference)/64;
                tree[current].solved=true;tree[current].value=value;}
            else{
                if(batching){
                    tree[current].pending=true;pending.push_back({path,length,p,moves});
                    for(int i=0;i<length;++i)++tree[path[i]].visits; // Temporary visits, removed before real backups.
                    if(int(pending.size())>=batchSize||(options.simulations&&analysis.nodes+pending.size()>=options.simulations))flush();
                    continue;
                }
                std::array<double,65> priors{};
                int evaluation;
                if(options.network){auto prediction=options.network->infer(p,moves,false);
                    evaluation=prediction.score;if(options.network->hasPolicy())priors=prediction.probabilities;}
                else evaluation=evaluate(p);
                expand(current,p,moves,priors);
                value=std::clamp(double(evaluation)/1000,-1.0,1.0);++analysis.evaluations;
            }
        }
        backup(path,length,value);
        if(options.simulations&&analysis.nodes+pending.size()>=options.simulations)flush();
    }
    flush();
    }catch(...){rollback();pending.clear();shared.reset(position);throw;}
    const auto& root=tree[0];analysis.exact=root.solved;
    if(root.firstChild>=0){
        int chosen=root.firstChild;
        if(outcome&&root.solved){
            for(int j=0;j<root.children;++j){int edge=root.firstChild+j,child=edges[edge].child;
                if(child>=0&&tree[child].solved&&-tree[child].value==root.value){chosen=edge;break;}}
        }
        auto valueOf=[&](int edge){int child=edges[edge].child;return child<0?0:-mean(tree[child]);};
        auto visitsOf=[&](int edge){int child=edges[edge].child;return child<0?std::uint64_t(0):tree[child].visits;};
        for(int j=0;j<root.children;++j){
            int index=root.firstChild+j;const Edge& edge=edges[index];
            const bool provenChoice=!(outcome&&root.solved)||(edge.child>=0&&tree[edge.child].solved);
            double q=valueOf(index),bq=valueOf(chosen);auto visits=visitsOf(index),bestVisits=visitsOf(chosen);
            bool better=root.solved?q>bq:visits>bestVisits||(visits==bestVisits&&q>bq);
            if(provenChoice&&(better||((root.solved?q==bq:visits==bestVisits&&q==bq)&&tie(position,edge.move,options.seed)>tie(position,edges[chosen].move,options.seed))))chosen=index;
            if(edge.move>=0){bool solved=edge.child>=0&&tree[edge.child].solved;
                analysis.candidates.push_back({edge.move,score(q,solved,outcome)});result.visits.push_back(visits);result.values.push_back(q);}
        }
        int child=edges[chosen].child;analysis.move=edges[chosen].move;analysis.score=score(valueOf(chosen),child>=0&&tree[child].solved,outcome);
    }else analysis.score=score(mean(root),root.solved,outcome);
    result.treeNodes=tree.size();result.meanDepth=analysis.nodes?totalDepth/analysis.nodes:0;
    analysis.elapsed=int(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count());return result;
}
}
