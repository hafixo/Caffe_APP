#ifndef ADAPTIVE_PROBABILISTIC_PRUNING_HPP_
#define ADAPTIVE_PROBABILISTIC_PRUNING_HPP_

#include <string>
#include <vector>
#include <map>
#define NUM_OF_WEIGHT_BUCKET 2
#define RATIO 0.5


namespace caffe {
using namespace std;

class APP {
public:
     APP() {};
    ~APP() {};

    /// --------------------------------
    /// pass params from solver.prototxt to layer
    static string prune_method;
    static string criteria;
    static int num_once_prune;
    static int prune_interval_begin;
    static int prune_interval_end;
    static int prune_iter_begin;
    static int prune_iter_end;
    static float recover_multiplier;
    static float range;
    static float rgamma;
    static float rpower;
    static float cgamma;
    static float cpower;
    static int iter_size;
    static float score_decay;
    
    static int inner_iter;
    static int step_;
    
    static map<string, int> layer_index;
    static int layer_cnt;
    
    static vector<float> num_pruned_col;
    static vector<int>   num_pruned_row;
    static vector<vector<bool> > IF_row_pruned;
    static vector<vector<vector<bool> > > IF_col_pruned; /// use 3 vectors because of: [layer, col, group]
    static vector<vector<float> > history_prob;
    static vector<int> iter_prune_finished;
    static vector<float> prune_ratio;
    static vector<float> delta;
    static vector<float> pruned_ratio;
    static vector<bool> IF_never_updated;
    
    static vector<int> filter_area;
    static vector<int> group;
    static vector<int> priority;
    
    static int num_log;
    static vector<vector<vector<float> > > log_weight;
    static vector<vector<vector<float> > > log_diff;
    static vector<vector<int> > log_index;


    /// --------------------------------
    static int window_size;  
    static float score_decay_rate;
    static bool use_score_decay;
    
    // When to Prune or Reg etc.
    static int when_to_col_reg;
    static float col_reg;
    static float diff_reg;
    
    // Decrease-Weight_Decay
    static int max_num_column_to_prune;

    // Selective Reg
    static float reg_decay;
    static bool use_selective_reg;
    static float selective_reg_cut_threshold;
    
    // Adaptive SPP
    static float loss; 
    static float loss_decay;
    static float Delta_loss_history;
    static float learning_speed;

    

    
}; 

}

#endif
