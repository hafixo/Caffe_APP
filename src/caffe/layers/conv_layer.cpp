#include <vector>
#include "caffe/layers/conv_layer.hpp"
#include "caffe/adaptive_probabilistic_pruning.hpp"
#include <cstdlib>
#include <cmath>
#define NSUM 50
#define SHOW_INTERVAL 20
#define SHOW_NUM 20

namespace caffe {
using namespace std;

template <typename Dtype>
void ConvolutionLayer<Dtype>::PruneSetUp(const PruneParameter& prune_param) {
    /// Basic setting
    const int count   = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    this->masks_.resize(count, 1); /// in test we still need masks and weight backup for PP
    this->weight_backup.resize(count);
    
    /// Get layer_index
    const string layer_name = this->layer_param_.name();
    if (this->phase_ == TRAIN) {
        if (APP::layer_index.count(layer_name) == 0) {
            ++ APP::layer_cnt;
            APP::layer_index[layer_name] = APP::layer_cnt;
        }
    } else { return; }
    
    const int L = APP::layer_index[layer_name];
    const string mthd = APP::prune_method;
    
    /// Note: the varibales below can ONLY be used in training.
    /// set up prune parameters
    this->prune_ratio = prune_param.prune_ratio();
    this->delta = prune_param.delta();
    this->pruned_ratio = 0;
    APP::prune_ratio.push_back(prune_param.prune_ratio());
    APP::delta.push_back(prune_param.delta());
    APP::pruned_ratio.push_back(0);
    
    /// info shared among different layers
    APP::num_pruned_col.push_back(0);
    APP::num_pruned_row.push_back(0);
    APP::IF_row_pruned.push_back( vector<bool>(num_row, false) );
    vector<bool> vec_tmp(this->group_, false);
    APP::IF_col_pruned.push_back( vector<vector<bool> >(num_col, vec_tmp) );
    
    if (mthd == "PPc") {
        APP::history_prob.push_back( vector<float>(num_col, 1) );
    } else {
        APP::history_prob.push_back( vector<float>(num_row, 1) );
    }
    APP::iter_prune_finished.push_back(INT_MAX);
    
    APP::filter_area.push_back(this->blobs_[0]->shape()[2] * this->blobs_[0]->shape()[3]);
    APP::group.push_back(this->group_);
    APP::priority.push_back(prune_param.priority());
    APP::IF_never_updated.push_back(true);
    
    /// Logging
    const int num_log = APP::num_log;
    Dtype rands[num_log];
    if (mthd == "FP" || mthd == "TP" || mthd == "PPr") {
        caffe_rng_uniform(num_log, (Dtype)0, (Dtype)(num_row - 1), rands);
    } else {
        caffe_rng_uniform(num_log, (Dtype)0, (Dtype)(num_col - 1), rands);
    }
    APP::log_index.push_back( vector<int>(num_log) ); /// the index of weights to be logged
    for (int i = 0; i < num_log; ++i) {
        APP::log_index[L][i] = int(rands[i]);
    }
    APP::log_weight.push_back( vector<vector<float> >(num_log) ); /// [layer, weight_index, weight-value-along-time]
    APP::log_diff.push_back( vector<vector<float> >(num_log) );   /// [layer, weight_index, diff-value-along-time]
    
    
    /// Pruning state info
    if (mthd == "PPc") {
        this->history_score.resize(num_col, 0);
    } else {
        this->history_score.resize(num_row, 0);
    }
    
    this->history_diff.resize(count, 0);
    this->blobs_[0]->mutable_cpu_second_diff = new Dtype[count];
    for (int i = 0; i < count; ++i) {
        this->blobs_[0]->mutable_cpu_second_diff[i] = 0;
    } // legacy
    
    if (num_col * this->prune_ratio > APP::max_num_column_to_prune) {
        APP::max_num_column_to_prune = num_col * this->prune_ratio;
    } // legacy
    
    cout << "=== Masks etc. Initialized." << endl;
}


template <typename Dtype>
bool ConvolutionLayer<Dtype>::IF_hppf() {
    /** IF_higher_priority_prune_finished 
    */
    bool IF_hppf = true;
    const int L = APP::layer_index[this->layer_param_.name()];
    for (int i = 0; i <= APP::layer_cnt; ++i) {
        if (APP::priority[i] < APP::priority[L] && APP::iter_prune_finished[i] != -1) {
            IF_hppf = false;
            break;
        }
    }
    return IF_hppf;
}

template <typename Dtype>
bool ConvolutionLayer<Dtype>::IF_alpf() {
    /** IF_all_layer_prune_finished
    */
    bool IF_alpf = true;
    for (int i = 0; i < APP::iter_prune_finished.size(); ++i) {
        if (APP::iter_prune_finished[i] == INT_MAX) {
            IF_alpf = false;
            break;
        }
    }
    return IF_alpf;
}

template <typename Dtype>
int ConvolutionLayer<Dtype>::GetPruneInterval() {
    const int x0 = APP::prune_iter_begin;
    const int x1 = APP::prune_iter_end;
    const int y0 = APP::prune_interval_begin;
    const int y1 = APP::prune_interval_end;
    const Dtype range = APP::range;
    
    const int interval_current = y0 - (y0 - y1) * 1.0 / (x1 - x0) * (APP::step_ - x0);
    if (range == 0) {
        return interval_current;
    } else {
        Dtype p;
        caffe_rng_uniform(1, Dtype(-range), Dtype(range), &p);
        return int((1 + p) * interval_current);
    }
    
}

template <typename Dtype> 
void ConvolutionLayer<Dtype>::TaylorPrune(const vector<Blob<Dtype>*>& top) {
    for (int i = 0; i < top.size(); ++i) {
        const Dtype* top_data = top[i]->cpu_data();
        const Dtype* top_diff = top[i]->cpu_diff();
        Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
        const int num_c = top[i]->shape()[1]; /// channel
        const int num_h = top[i]->shape()[2]; /// height
        const int num_w = top[i]->shape()[3]; /// width
        const int count = this->blobs_[0]->count();
        const int num_row = this->blobs_[0]->shape()[0];
        const int num_col = count / num_row;
        const int L = APP::layer_index[this->layer_param_.name()];

        typedef std::pair<Dtype, int> mypair;
        vector<mypair> fm_score(num_c); /// feature map score
        for (int c = 0; c < num_c; ++c) {
            fm_score[c].second = c;
            fm_score[c].first  = 0;
        }
        for (int n = 0; n < this->num_; ++n) {
            for (int c = 0; c < num_c; ++c) {
                for (int i = 0; i < num_h * num_w; ++i) {
                    fm_score[c].first += fabs(top_diff[n * num_c * num_w * num_h + c * num_w * num_h + i] 
                                            * top_data[n * num_c * num_w * num_h + c * num_w * num_h + i]);                          
                }
            }
        }
        for (int c = 0; c < num_c; ++c) {
            if (APP::IF_row_pruned[L][c]) {
                fm_score[c].first = INT_MAX;
            }
        }
        sort(fm_score.begin(), fm_score.end());
        int num_once_prune = 1;
        if (APP::num_once_prune > 1) { num_once_prune = APP::num_once_prune; }
        for (int i = 0; i < num_once_prune; ++i) {
            const int c = fm_score[i].second;
            for (int j = 0; j < num_col; ++j) {
                muweight[c * num_col + j] = 0; /// Seems don't work
                this->masks_[c * num_col + j] = 0;
            }
            APP::IF_row_pruned[L][c] = true;
            ++ APP::num_pruned_row[L];
        }
    }
}
    

template <typename Dtype> 
void ConvolutionLayer<Dtype>::ProbPruneCol() {
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    const string layer_name = this->layer_param_.name();
    const int num_col_to_prune_ = ceil((this->prune_ratio + this->delta) * num_col); /// a little bit higher goal
    const int iter_size = APP::iter_size;
    const Dtype rgamma = APP::rgamma;
    const Dtype rpower = APP::rpower;
    const Dtype cgamma = APP::cgamma;
    const Dtype cpower = APP::cpower;
    const int L = APP::layer_index[layer_name];
    
    
    /// Calculate history score
    typedef std::pair<Dtype, int> mypair;
    vector<mypair> col_score(num_col);
    for (int j = 0; j < num_col; ++j) {
        col_score[j].second = j; /// index
        Dtype score = 0;
        for (int i = 0; i < num_row; ++i) {
            score += fabs(muweight[i * num_col +j]);
        }
        this->history_score[j] = APP::score_decay * this->history_score[j] + score;
        col_score[j].first = this->history_score[j];
        if (APP::IF_col_pruned[L][j][0]) { col_score[j].first = INT_MAX; } /// make the pruned columns "float" up
    }
    sort(col_score.begin(), col_score.end());
    
    /// Recover the best columns, according to some probabilities
    Dtype p_recover;
    caffe_rng_uniform(1, (Dtype)0, (Dtype)1, &p_recover);
    if (pow(rgamma + 0.00027 * APP::step_, rpower) > p_recover * iter_size) {
    /// const int recover_interval = APP::recover_multiplier * GetPruneInterval();
    /// if (APP::step_ % recover_interval == 0) {

        /// Print and check
        cout << "recover prob: " << layer_name << "  step: " << APP::step_ << endl;
        cout << " score: ";   for (int j = 0; j < SHOW_NUM; ++j) { cout << col_score[j].first  << " "; }
        cout << "\ncolumn: "; for (int j = 0; j < SHOW_NUM; ++j) { cout << col_score[j].second << " "; }
        cout << "\n  prob: "; for (int j = 0; j < SHOW_NUM; ++j) { cout << APP::history_prob[L][col_score[j].second] << " "; }
        cout << "\n";                    
        
        for (int j = num_col_to_prune_ - APP::num_pruned_col[L] - 1; 
                 j < num_col - APP::num_pruned_col[L]; ++j) {
            const int col_of_rank_j = col_score[j].second;
            cout << "recover col: " << col_of_rank_j 
                 << "  its prob: " << APP::history_prob[L][col_of_rank_j] 
                 << "  step: " << APP::step_ << endl;
            APP::history_prob[L][col_of_rank_j] = 1;
        }
    }

    /// Update history_prob, according to some probabilities
    /// if (std::min(Dtype(APP::learning_speed), (Dtype)0.004) * 4 > p_prune) {  
    Dtype p_prune;
    caffe_rng_uniform(1, (Dtype)0, (Dtype)1, &p_prune);
    if (pow(cgamma + 0.0008 * APP::step_, cpower) > p_prune * iter_size) {
    /// if (APP::step_ % GetPruneInterval() == 0) {
    
        /// Print and check
        cout << "update prob: " << layer_name << " step: " << APP::step_ << endl;
        cout << " score: ";   for (int j = 0; j < SHOW_NUM; ++j) { cout << col_score[j].first  << " "; }
        cout << "\ncolumn: "; for (int j = 0; j < SHOW_NUM; ++j) { cout << col_score[j].second << " "; }
        cout << "\n  prob: "; for (int j = 0; j < SHOW_NUM; ++j) { cout << APP::history_prob[L][col_score[j].second] << " "; }
        cout << "\n";
    
        /// Calculate functioning probability of each weight
        const Dtype AA = 0.05; 
        const Dtype aa = 0.0041;
        const int group = APP::group[L];
        const Dtype alpha = -log(aa/AA) / (num_col_to_prune_ - APP::num_pruned_col[L] - 1);  /// adjust alpha according to the remainder of cloumns
        for (int j = 0; j < num_col_to_prune_ - APP::num_pruned_col[L]; ++j) {               /// note the range of j: only undermine those not-good-enough columns
            const int col_of_rank_j = col_score[j].second;
            APP::history_prob[L][col_of_rank_j] = std::max(APP::history_prob[L][col_of_rank_j] - AA * exp(-j * alpha), (Dtype)0);
            
            if (APP::history_prob[L][col_of_rank_j] == 0) {
                APP::num_pruned_col[L] += 1;
                for (int g = 0; g < group; ++g) {
                    APP::IF_col_pruned[L][col_of_rank_j][g] = true;
                }
                for (int i = 0; i < num_row; ++i) { 
                    muweight[i * num_col + col_of_rank_j] = 0; 
                } /// once pruned, zero out weights
            }
        } 
    }

    /// With probability updated, generate masks
    Dtype rands[num_col];
    caffe_rng_uniform(num_col, (Dtype)0, (Dtype)1, rands);
    for (int i = 0; i < count; ++i) {
        this->masks_[i] = rands[i % num_col] < APP::history_prob[L][i % num_col] ? 1 : 0; /// generate masks
    }              
    for (int i = 0; i < count; ++i) { this->weight_backup[i] = muweight[i]; }
    this->IF_restore = true;
    for (int i = 0; i < count; ++i) { muweight[i] *= this->masks_[i]; } /// do pruning
}

template <typename Dtype> 
void ConvolutionLayer<Dtype>::ProbPruneRow() {
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    const string layer_name = this->layer_param_.name();
    const int num_row_to_prune_ = ceil((this->prune_ratio + this->delta) * num_row); /// a little bit higher goal
    const int iter_size = APP::iter_size;
    const Dtype rgamma = APP::rgamma;
    const Dtype rpower = APP::rpower;
    const Dtype cgamma = APP::cgamma;
    const Dtype cpower = APP::cpower;
    const int L = APP::layer_index[layer_name];
    
    
    /// Calculate history score
    typedef std::pair<Dtype, int> mypair;
    vector<mypair> row_score(num_row);
    for (int i = 0; i < num_row; ++i) {
        row_score[i].second = i; /// index
        Dtype score = 0;
        for (int j = 0; j < num_row; ++j) {
            score += fabs(muweight[i * num_col +j]);
        }
        this->history_score[i] = APP::score_decay * this->history_score[i] + score;
        row_score[i].first = this->history_score[i];
        if (APP::IF_row_pruned[L][i]) { row_score[i].first = INT_MAX; } /// make the pruned rows "float" up
    }
    sort(row_score.begin(), row_score.end());
    
    /// Recover the best columns, according to some probabilities
    Dtype p_recover;
    caffe_rng_uniform(1, (Dtype)0, (Dtype)1, &p_recover);
    if (pow(rgamma + 0.00027 * APP::step_, rpower) > p_recover * iter_size) {

        /// Print and check
        cout << "recover prob: " << layer_name << "  step: " << APP::step_ << endl;
        cout << " score: ";   for (int i = 0; i < num_row; ++i) { cout << row_score[i].first  << " "; }
        cout << "\ncolumn: "; for (int i = 0; i < num_row; ++i) { cout << row_score[i].second << " "; }
        cout << "\n  prob: "; for (int i = 0; i < num_row; ++i) { cout << APP::history_prob[L][row_score[i].second] << " "; }
        cout << "\n";
        
        for (int i = num_row_to_prune_ - APP::num_pruned_row[L] - 1; 
                 i < num_row - APP::num_pruned_row[L]; ++i) {
            const int row_of_rank_i = row_score[i].second;
            cout << "recover row: " << row_of_rank_i
                 << "  its prob: " << APP::history_prob[L][row_of_rank_i] 
                 << "  step: " << APP::step_ << endl;
            APP::history_prob[L][row_of_rank_i] = 1;
        }
    }

    /// Update history_prob, according to some probabilities
    Dtype p_prune;
    caffe_rng_uniform(1, (Dtype)0, (Dtype)1, &p_prune);
    if (pow(cgamma + 0.0008 * APP::step_, cpower) > p_prune * iter_size) {
    
        /// Print and check
        cout << "update prob: " << layer_name << "  step: " << APP::step_ << endl;
        cout << " score: ";   for (int i = 0; i < num_row; ++i) { cout << row_score[i].first  << " "; }
        cout << "\ncolumn: "; for (int i = 0; i < num_row; ++i) { cout << row_score[i].second << " "; }
        cout << "\n  prob: "; for (int i = 0; i < num_row; ++i) { cout << APP::history_prob[L][row_score[i].second] << " "; }
        cout << "\n";   
    
        /// Calculate functioning probability of each weight
        const Dtype AA = 0.05; 
        const Dtype aa = 0.0041;
        const Dtype alpha = -log(aa/AA) / (num_row_to_prune_ - APP::num_pruned_row[L] - 1);  /// adjust alpha according to the remainder of rows
        for (int i = 0; i < num_row_to_prune_ - APP::num_pruned_row[L]; ++i) {               /// note the range of j: only undermine those not-good-enough rows
            const int row_of_rank_i = row_score[i].second;
            APP::history_prob[L][row_of_rank_i] = std::max(APP::history_prob[L][row_of_rank_i] - AA * exp(-i * alpha), (Dtype)0);
            
            if (APP::history_prob[L][row_of_rank_i] == 0) {
                ++ APP::num_pruned_row[L];                
                APP::IF_row_pruned[L][row_of_rank_i] = true;
                for (int j = 0; j < num_col; ++j) { 
                    muweight[row_of_rank_i * num_col + j] = 0; 
                } /// once pruned, zero out weights
            }
        }
    }

    // With probability updated, generate masks
    Dtype rands[num_row];
    caffe_rng_uniform(num_row, (Dtype)0, (Dtype)1, rands);
    for (int i = 0; i < count; ++i) {
        this->masks_[i] = rands[i / num_col] < APP::history_prob[L][i / num_col] ? 1 : 0; /// generate masks
        this->weight_backup[i] = muweight[i]; /// backup weights to restore
    }              
    this->IF_restore = true;
    for (int i = 0; i < count; ++i) { muweight[i] *= this->masks_[i]; } /// do pruning
}

template <typename Dtype> 
void ConvolutionLayer<Dtype>::CleanWorkForPP() {
    /// Once the pruning ratio reached, set all the masks of non-zero prob to 1 and adjust their weights.
    /// Get into here ONLY ONCE.
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;  
    const int L = APP::layer_index[this->layer_param_.name()];
    
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    for (int i = 0; i < count; ++i) {
        if (APP::history_prob[L][i % num_col] > 0) {
            muweight[i] *= APP::history_prob[L][i % num_col];
            APP::history_prob[L][i % num_col] = 1;
        }
    }
}

template <typename Dtype> 
void ConvolutionLayer<Dtype>::UpdateNumPrunedRow() {
    const int L = APP::layer_index[this->layer_param_.name()];
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int filter_area_next_layer = APP::filter_area[L+1]; 
    const int num_col = count / num_row;
    
    cout << "   conv" << L + 1 << " in UpdateNumPrunedRow" << endl;
    for (int i = 0; i < num_row; ++i) {
        if (!APP::IF_row_pruned[L][i]) {
            const int i_ = i % (num_row / APP::group[L + 1]);
            const int g  = i / (num_row / APP::group[L + 1]);
            bool IF_consecutive_pruned = true;
            for (int j = i_ * filter_area_next_layer; j < (i_ + 1) * filter_area_next_layer; ++j) {
                if (!APP::IF_col_pruned[L + 1][j][g]) { 
                    IF_consecutive_pruned = false; 
                    break;
                }
            }
            if (IF_consecutive_pruned) {
                for (int j = 0; j < num_col; ++j) {
                    muweight[i * num_col + j] = 0;
                    this->masks_[i * num_col + j] = 0;
                }
                APP::IF_row_pruned[L][i] = true;
                ++ APP::num_pruned_row[L];
                cout << "   conv" << L + 1 << " prune a row successfully: " << i << endl;
            }
        }
    }
    APP::IF_never_updated[L] = false;
}

template <typename Dtype> 
void ConvolutionLayer<Dtype>::UpdateNumPrunedCol() {
    const int L = APP::layer_index[this->layer_param_.name()];
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int filter_area = this->blobs_[0]->shape()[2] * this->blobs_[0]->shape()[3];
    const int num_col = count / num_row;

    const int group =  APP::group[L];
    const int num_chl_per_g = this->blobs_[0]->shape()[1] / group; /// num channel per group
    const int num_row_per_g = this->blobs_[0]->shape()[0] / group;
    
    cout << "   conv" << L+1 << " in UpdateNumPrunedCol" << endl;
    for (int j = 0; j < num_col; ++j) {
        const int chl_ix = j / filter_area; /// channel index
        for (int g = 0; g < group; ++g) {
            const bool cond1 = !(APP::IF_col_pruned[L][j][g]);
            const bool cond2 =  APP::IF_row_pruned[L - 1][chl_ix + g * num_chl_per_g];
            if (cond1 && cond2) {
                for (int i = g * num_row_per_g ; i < (g+1) * num_row_per_g; ++i) {
                    muweight[i * num_col + j] = 0;
                    this->masks_[i * num_col + j] = 0;
                }
                APP::IF_col_pruned[L][j][g] = true;
                APP::num_pruned_col[L] += 1.0 / group;
                cout << "   conv" << L+1 << " prune a col_group successfully: " << j << "-" << g << endl;
            }
        }
    }
    APP::IF_never_updated[L] = false;
}

template <typename Dtype> 
void ConvolutionLayer<Dtype>::FilterPrune() {
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    const int L = APP::layer_index[this->layer_param_.name()];
     
    typedef pair<Dtype, int> mypair;
    vector<mypair> row_score(num_row);
    for (int i = 0; i < num_row; ++i) {
        row_score[i].second = i; /// index 
        if (APP::IF_row_pruned[L][i]) { 
            row_score[i].first = INT_MAX; /// make those pruned row "float" up
            continue;
        } 
        row_score[i].first  = 0; /// score
        for (int j = 0; j < num_col; ++j) {
            row_score[i].first += fabs(muweight[i * num_col +j]);
        }
    }
    sort(row_score.begin(), row_score.end()); /// in ascending order
    for (int i = 0; i < APP::num_once_prune; ++i) {
        for (int j = 0; j < num_col; ++j) {
            muweight[row_score[i].second * num_col + j] = 0;
            this->masks_[row_score[i].second * num_col + j] = 0;
        }
        APP::IF_row_pruned[L][row_score[i].second] = true;
        ++ APP::num_pruned_row[L];
    }
}


template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    /// i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::ComputeBlobMask(float ratio) {
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    const Dtype* weight = this->blobs_[0]->cpu_data();
    const string layer_name = this->layer_param_.name();
    const int L = APP::layer_index[layer_name];
    const int group = APP::group[L];
    const int num_row_per_g = num_row / group;

    Dtype num_pruned_col = 0;
    int   num_pruned_row = 0;
    for (int j = 0; j < num_col; ++j) {
        for (int g = 0; g < group; ++g) {
            Dtype sum = 0;
            for (int i = g * num_row_per_g; i < (g+1) * num_row_per_g; ++i) { 
                sum += fabs(weight[i * num_col + j]); 
            }
            if (sum == 0) { 
                num_pruned_col += 1.0 / group;
                APP::IF_col_pruned[L][j][g] = true;
                for (int i = g * num_row_per_g; i < (g+1) * num_row_per_g; ++i) { 
                    this->masks_[i * num_col + j] = 0; 
                }
            }
        }
    }
    for (int i = 0; i < num_row; ++i) { 
        Dtype sum = 0;
        for (int j = 0; j < num_col; ++j) { 
            sum += fabs(weight[i * num_col + j]); 
        }
        if (sum == 0) {
            ++ num_pruned_row;
            APP::IF_row_pruned[L][i] = true;
            for (int j = 0; j < num_col; ++j) { 
                this->masks_[i * num_col + j] = 0; 
            }
        }
    }
    APP::num_pruned_col[L] = num_pruned_col;
    APP::num_pruned_row[L] = num_pruned_row;
    this->pruned_ratio = 1 - (1 - num_pruned_col / num_col) * (1 - num_pruned_row * 1.0 / num_row);
    if (this->pruned_ratio >= this->prune_ratio) {
        APP::iter_prune_finished[L] = APP::step_ - 1;
    }
    LOG(INFO) << "    Masks restored, num_pruned_col = " << num_pruned_col
              << "  num_pruned_row = " << num_pruned_row
              << "  pruned_ratio = " << this->pruned_ratio
              << "  prune_ratio = " << this->prune_ratio;
}


template <typename Dtype>
Dtype ConvolutionLayer<Dtype>::normal_random() {
    static Dtype V1, V2, S;
    static int phase = 0;
    Dtype X;
    if (phase == 0) {
        do {
            Dtype U1 = (Dtype) rand() / RAND_MAX;
            Dtype U2 = (Dtype) rand() / RAND_MAX;
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while (S >= 1 || S == 0);  /// loop until 0<S<1
        X = V1 * sqrt(-2 * log(S) / S);
    } else {
        X = V2 * sqrt(-2 * log(S) / S);
    }
    phase = 1 - phase;
    return X * 0.05;
}


template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data(); 

    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);

      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}



template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* weight = this->blobs_[0]->cpu_data(); /// weight用来计算底层的梯度dx = dz * w
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();

  for (int i = 0; i < top.size(); ++i) { /// 对于top层中的每个神经元
    const Dtype* top_diff = top[i]->cpu_diff(); /// top_diff是dz
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();

    /// Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) { /// num_是在base_conv中定义的
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }


    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        /// gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff); /// calculate weight_diff for this layer
        }
        
        /// gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_); /// dx = dz * w
        }
      }
    }
  }

}


#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  /// namespace caffe
