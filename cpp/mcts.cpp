#include "mcts.hpp"
#include "nnue.hpp"
#include "pattern.hpp"
#include <cmath>
#include <limits>
#include <memory>

namespace othello {
namespace {
struct Node {
    Position position;
    std::uint64_t visits=0;
    double total=0,value=0;
    int firstChild=-1,children=0,move=-1;
    bool solved=false;
};
struct Edge {int child=-1,move=-1;double prior=0;};
bool samePosition(const Position& a,const Position& b){return a.hash==b.hash&&a.black==b.black&&a.white==b.white&&a.turn==b.turn;}
struct PatternEntry {nnue::Pattern::State state;std::uint64_t generation=0;int node=-1;};
struct Cache {
    std::vector<PatternEntry> patternStates;
    std::uint64_t patternGeneration=1;
    void invalidatePattern(){++patternGeneration;}
    std::vector<Node> tree;
    std::vector<Edge> edges;
    std::vector<double> priors; // allocated only for the pattern-policy experiment
    std::unique_ptr<nnue::Network> evaluator;
    bool policy=false;
    void reset(const Position& p){invalidatePattern();tree.clear();edges.clear();priors.clear();Node n;n.position=p;tree.push_back(n);}
    std::size_t prepare(const Position& p,const nnue::Network* network,bool usePolicy){
        const bool same=policy==usePolicy&&(network&&evaluator?network->sameParameters(*evaluator):!network&&!evaluator);
        if(!same){invalidatePattern();tree.clear();edges.clear();priors.clear();evaluator=network?network->snapshot():nullptr;}
        policy=usePolicy;
        if(!tree.empty()&&samePosition(tree[0].position,p))return tree.size();
        int root=-1;
        for(std::size_t i=0;i<tree.size();++i)if(samePosition(tree[i].position,p)){root=int(i);break;}
        if(root<0){reset(p);return 0;}
        invalidatePattern();
        std::vector<Node> kept;std::vector<Edge> keptEdges;std::vector<double> keptPriors;
        kept.reserve(tree.size());keptEdges.reserve(edges.size());kept.push_back(tree[root]);
        for(std::size_t i=0;i<kept.size();++i){
            int old=kept[i].firstChild,n=kept[i].children;
            if(old>=0){kept[i].firstChild=int(keptEdges.size());
                for(int j=0;j<n;++j){Edge edge=edges[old+j];
                    if(edge.child>=0){int child=edge.child;edge.child=int(kept.size());kept.push_back(tree[child]);}
                    keptEdges.push_back(edge);
                    if(!priors.empty())keptPriors.push_back(priors[old+j]);
                }
            }
        }
        tree.swap(kept);edges.swap(keptEdges);priors.swap(keptPriors);return tree.size();
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
void clearMctsCache(){auto& c=cache();c.tree.clear();c.edges.clear();c.priors.clear();c.evaluator.reset();c.invalidatePattern();}
MctsResult mcts(const Position& position,MctsOptions options){
    const bool outcome=options.network&&options.network->valueTarget==nnue::ValueTarget::Outcome;
    if(options.milliseconds<0||options.milliseconds>60000||(!options.milliseconds&&!options.simulations)
       ||(options.patternCacheEntries&&(options.patternCacheEntries<64||options.patternCacheEntries>16384||(options.patternCacheEntries&(options.patternCacheEntries-1))))
       ||!std::isfinite(options.exploration)||options.exploration<=0||options.maxNodes<65||options.maxNodes>16777216)
        throw std::invalid_argument("Invalid MCTS budget/exploration/node limit");
    const auto start=std::chrono::steady_clock::now();const auto deadline=start+std::chrono::milliseconds(options.milliseconds);
    const bool learnedPolicy=options.usePolicy&&options.network&&options.network->pattern;
    MctsResult result;auto& shared=cache();result.reusedNodes=shared.prepare(position,options.network,options.usePolicy);
    auto& tree=shared.tree;auto& edges=shared.edges;
    if(tree.size()>options.maxNodes){shared.reset(position);result.reusedNodes=0;}
    const std::size_t stateEntries=learnedPolicy?options.patternCacheEntries:0;
    if(shared.patternStates.size()!=stateEntries){
        shared.patternStates=std::vector<PatternEntry>(stateEntries);shared.invalidatePattern();
    }
    result.patternCacheBytes=shared.patternStates.capacity()*sizeof(PatternEntry);
    auto& analysis=result.analysis;double totalDepth=0;
    while(!tree[0].solved&&(!options.simulations||analysis.nodes<options.simulations)){
        if(analysis.nodes&&options.milliseconds&&std::chrono::steady_clock::now()>=deadline)break;
        std::array<int,130> path{};int length=1,current=0;path[0]=0;
        while(tree[current].firstChild>=0&&!tree[current].solved){
            const auto& parent=tree[current];int chosen=-1;double best=-std::numeric_limits<double>::infinity();
            const bool learned=options.usePolicy&&options.network&&options.network->hasPolicy();
            const double exploration=options.exploration*std::sqrt(double(std::max<std::uint64_t>(1,parent.visits)));
            for(int j=0;j<parent.children;++j){
                int index=parent.firstChild+j;const Edge& edge=edges[index];
                std::uint64_t visits=edge.child<0?0:tree[edge.child].visits;
                const double prior=learned?exploration*edge.prior:exploration/parent.children;
                double utility;
                if(learnedPolicy){
                    // Preserve the Pattern experiment's finite first-play value.
                    const double q=visits?-mean(tree[edge.child]):mean(parent)-0.15;
                    utility=q+exploration*shared.priors[index]/(1+visits);
                }else utility=visits?-mean(tree[edge.child])+prior/(1+visits):std::numeric_limits<double>::infinity();
                bool preferred=chosen>=0&&learned&&utility==best&&edge.prior!=edges[chosen].prior;
                if(chosen<0||utility>best||(utility==best&&(preferred?edge.prior>edges[chosen].prior:
                    tie(parent.position,edge.move,options.seed)>tie(parent.position,edges[chosen].move,options.seed)))){best=utility;chosen=index;}
            }
            if(length>=int(path.size()))throw std::runtime_error("MCTS path exceeds game length");
            if(edges[chosen].child<0){
                if(tree.size()>=options.maxNodes){result.memoryLimited=true;break;}
                Node child;child.move=edges[chosen].move;child.position=play(tree[current].position,child.move);
                edges[chosen].child=int(tree.size());tree.push_back(child);
            }
            current=edges[chosen].child;path[length++]=current;
        }
        if(result.memoryLimited)break;
        double value;
        if(tree[current].solved)value=tree[current].value;
        else{
            const Position p=tree[current].position;Bits moves=legal(p);bool terminal=!moves&&!legal(p.opponent(),p.own());
            if(terminal){int difference=count(p.own())-count(p.opponent());
                value=outcome?double((difference>0)-(difference<0)):double(difference)/64;
                tree[current].solved=true;tree[current].value=value;}
            else{
                tree[current].firstChild=int(edges.size());tree[current].children=moves?count(moves):1;
                if(learnedPolicy){
                    const auto& model=*options.network->pattern;
                    nnue::Pattern::Prediction patternPrediction;
                    if(options.patternCacheEntries){
                        PatternEntry* ancestor=nullptr;int ancestorDepth=0;
                        for(int i=length-1;i>=0;--i){
                            ++result.patternCacheLookups;
                            auto& entry=shared.patternStates[std::size_t(path[i])&(options.patternCacheEntries-1)];
                            if(entry.generation==shared.patternGeneration&&entry.node==path[i]){ancestor=&entry;ancestorDepth=i;break;}
                        }
                        nnue::Pattern::State state;
                        if(ancestor){unsigned changed=0;state=model.updateState(ancestor->state,p,&changed);
                            ++result.patternUpdates;++result.patternCacheHits;result.patternChangedPatterns+=changed;result.patternAncestorDistance+=length-1-ancestorDepth;
                        }else{state=model.refreshState(p);++result.patternRefreshes;}
                        auto& entry=shared.patternStates[std::size_t(current)&(options.patternCacheEntries-1)];
                        entry.state=std::move(state);entry.node=current;entry.generation=shared.patternGeneration;
#if defined(OTELLO_PATTERN_LEGAL_POLICY)
                        patternPrediction=model.predictLegal(entry.state,p.turn,moves);
#else
                        patternPrediction=model.predict(entry.state,p.turn);
#endif
                    }else{
#if defined(OTELLO_PATTERN_LEGAL_POLICY)
                        patternPrediction=model.predictLegal(p,moves);
#else
                        patternPrediction=model.predict(p);
#endif
                        ++result.patternRefreshes;
                    }
                    if(!moves)edges.push_back({-1,-1});
                    else while(moves){int move=first(moves);moves&=moves-1;edges.push_back({-1,move});}
                    shared.priors.resize(edges.size());
                    const auto& logits=patternPrediction.logits;int begin=tree[current].firstChild,n=tree[current].children;double maximum=-1e30,total=0;
                    for(int j=0;j<n;++j)maximum=std::max(maximum,edges[begin+j].move<0?0:logits[edges[begin+j].move]);
                    for(int j=0;j<n;++j){const auto& e=edges[begin+j];auto& prior=shared.priors[begin+j];prior=std::exp((e.move<0?0:logits[e.move])-maximum);total+=prior;}
                    for(int j=0;j<n;++j)shared.priors[begin+j]/=total;
                    value=std::round(1000*std::tanh(double(patternPrediction.values[0])/256))/1000;++analysis.evaluations;
                }else{
                    std::array<double,65> priors{};
                    int evaluation;
                    if(options.network){auto prediction=options.network->infer(p,moves,false);
                        evaluation=prediction.score;if(options.network->hasPolicy())priors=prediction.probabilities;}
                    else evaluation=evaluate(p);
                    if(!moves)edges.push_back({-1,-1,1});
                    else while(moves){int move=first(moves);moves&=moves-1;edges.push_back({-1,move,priors[move]});}
                    value=std::clamp(double(evaluation)/1000,-1.0,1.0);++analysis.evaluations;
                }
            }
        }
        for(int i=length-1;i>=0;--i){auto& n=tree[path[i]];n.total+=value;++n.visits;solve(path[i],shared,outcome);value=-value;}
        ++analysis.nodes;analysis.depth=std::max(analysis.depth,length-1);totalDepth+=length-1;
    }
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
