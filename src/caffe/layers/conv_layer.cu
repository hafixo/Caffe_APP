#include <vector>
#include "caffe/layers/conv_layer.hpp"
#include "caffe/adaptive_probabilistic_pruning.hpp"
#include <ctime>

using namespace std;
namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
          
    /// ADDED BY WANGHUAN -----------------------------------
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    const string layer_name = this->layer_param_.name();
    const string mthd = APP::prune_method;
    const int L = APP::layer_index[layer_name];
    this->IF_restore = false;
    
    #ifdef ShowTimingLog
    cout << layer_name << " forward start timing" << endl;
    clock_t t1 = clock();
    #endif

    /// IF_prune
    const bool IF_want_prune  = mthd != "None" && APP::prune_ratio[L] > 0; // if you want to prune, you must specify a meaningful prune_method and give a positive prune_ratio
    const bool IF_been_pruned = APP::pruned_ratio[L] > 0; // for a pruned layer, continue to prune
    const bool IF_enough_iter = APP::step_ >= APP::prune_begin_iter+1; // for a raw layer, if iter is enough, then prune
    const bool IF_prune = IF_want_prune && (IF_been_pruned || IF_enough_iter);
    
    if (this->phase_ == TRAIN && APP::inner_iter == 0) {
        // For a layer which doesn't want to prune, it still should UpdateNumPrunedCol/Row because of neighbour layer
        if (mthd != "None" && (IF_been_pruned || IF_enough_iter)) { 
            if (APP::IF_update_row_col && APP::IF_update_row_col_layer[L]) {
                // Note that, UpdateNumPrunedRow/Col before pruning, so that when calculating score, the zombie weights will not be counted.
                // The last conv and last fc layer need not updating num of pruned row.
                // In fact, the last conv should be updated row and the first fc should be updated col, but for simplicity, which are ignored for now.
                if (APP::prune_unit == "Col" && L != APP::conv_layer_cnt-1) { 
                    if (APP::step_-1 - APP::iter_prune_finished[L+1] <= 1) {
                        //UpdateNumPrunedRow();
                    }
                } else if (APP::prune_unit == "Row" && mthd != "TP_Row" && APP::pruned_rows.size()) {
                    UpdateNumPrunedCol();
                } /// Note we don't update column for TP, because their method didn't mention this.
                UpdatePrunedRatio();
            }
            
            // check if prune finished, get into here ONLY once
            if (APP::iter_prune_finished[L] == INT_MAX) {
                Dtype pruned_ratio;
                if (APP::prune_unit == "Weight")   { pruned_ratio = APP::pruned_ratio[L];     }
                else if (APP::prune_unit == "Row") { pruned_ratio = APP::pruned_ratio_row[L]; }
                else if (APP::prune_unit == "Col") { pruned_ratio = APP::pruned_ratio_col[L]; }
                const bool layer_finish     = pruned_ratio >= APP::prune_ratio[L]; /// layer pruning target achieved
                const bool net_finish_speed = APP::IF_speedup_achieved;   /// net pruning target of speed achieved
                const bool net_finish_param = APP::IF_compRatio_achieved; /// net pruning target of compression achieved
                
                if (layer_finish || net_finish_speed || net_finish_param) {
                    APP::iter_prune_finished[L] = APP::step_ - 1;
                    if (APP::prune_coremthd.substr(0,2) == "PP") { CleanWorkForPP(); } // last time, do some clean work
                    
                    // print when finished
                    char rlayer[10], rrow[10], rcol[10];
                    sprintf(rlayer, "%6.4f", APP::pruned_ratio[L]);
                    sprintf(rrow,   "%6.4f", APP::pruned_ratio_row[L]);
                    sprintf(rcol,   "%6.4f", APP::pruned_ratio_col[L]);
                    cout << layer_name << " prune finished!" 
                         << "  step: " << APP::step_
                         << "  net speedup: " << APP::speedup
                         << "  net compRatio: " << APP::compRatio
                         << "  pruned_ratio: " << rlayer
                         << "  pruned_ratio_row: " << rrow
                         << "  pruned_ratio_col: " << rcol 
                         << "  prune_ratio: " << APP::prune_ratio[L] << endl;
                    IF_alpf();
                }
            }
        }
        
        // Print and check, before update probs
        // put this outside, to print even when we do not prune
        if (L == APP::show_layer && APP::step_ % APP::show_interval == 0) {
            Print(L, 'f');
        }

        // Update masks and apply masks
        if (IF_prune && APP::iter_prune_finished[L] == INT_MAX) {
            if (mthd == "FP_Row" && (APP::step_ - 1) % APP::prune_interval == 0) {
                FilterPrune(); 
            } else if (mthd == "PP_Col" && IF_hppf()) {
                ProbPruneCol(APP::prune_interval);
            } else if (mthd == "PP_Row" && IF_hppf()) {
                ProbPruneRow(APP::prune_interval);
            } else if (APP::prune_coremthd.substr(0,3) == "Reg") {
                PruneMinimals();
            } else {
                LOG(INFO) << "Wrong: unknown prune_method";
                exit(1);
            }
            UpdatePrunedRatio();
            if (L == APP::conv_layer_cnt - 1) { // To avoid the first fc from updating col
                APP::pruned_rows.clear();
            }
        }
        
        // Print weight magnitude
        if (APP::num_log > 0) {
            if (APP::prune_unit == "Col") {
                cout << "ave-magnitude_col " << APP::step_ << " " << layer_name << ":";
                for (int j = 0; j < num_col; ++j) {
                    Dtype sum = 0;
                    for (int i = 0; i < num_row; ++i) {
                        sum += fabs(muweight[i*num_col + j]);
                    }
                    cout << " " << sum;
                }
                cout << endl;
            } else if (APP::prune_unit == "Row") {
                cout << "ave-magnitude_row " << APP::step_ << " " << layer_name << ":";
                for (int i = 0; i < num_row; ++i) {
                    Dtype sum = 0;
                    for (int j = 0; j < num_col; ++j) {
                        sum += fabs(muweight[i*num_col + j]);
                    }
                    cout << " " << sum;
                }
                cout << endl;
            }
        }
        
        // Summary print 
        int cnt_negative = 0;
        for (int i = 0; i < count; ++i) {
            if (muweight[i] < 0) {
                ++ cnt_negative;
            }
        }
        if (APP::step_ % APP::show_interval == 0 && IF_prune && mthd != "None" && L < APP::show_num_layer) {
            cout << layer_name << "  IF_prune: " << IF_prune 
                 << "  pruned_ratio: " << APP::pruned_ratio[L];
            cout << "  pruned_ratio_row: " << APP::num_pruned_row[L] * 1.0 / num_row << "(" << APP::num_pruned_row[L] << ")"
                 << "  pruned_ratio_col: " << APP::num_pruned_col[L] * 1.0 / num_col << "(" << APP::num_pruned_col[L] << ")";
            cout << "  prune_ratio: "  << APP::prune_ratio[L] 
	         << "  num_negative: " << cnt_negative << "(" << cnt_negative*1.0/count << ")" << endl;
        }
        
    } else if (this->phase_ == TEST && IF_prune && APP::iter_prune_finished[L] == INT_MAX && APP::prune_coremthd.substr(0,2) == "PP") {
        Dtype rands[count / 10];
        for (int i = 0; i < count; ++i) {
            if (i % (count/10) == 0) {
                caffe_rng_uniform(count/10, (Dtype)0, (Dtype)1, rands);
            }
            APP::masks[L][i] = rands[i%(count/10)] < APP::history_prob[L][i] ? 1 : 0;
            this->weight_backup[i] = muweight[i];
            muweight[i] *= APP::masks[L][i];
        }
        this->IF_restore = true;
    }
    
    // Quantization ------------------------------
    if (APP::num_bit) {
        const int n = pow(2, APP::num_bit) - 1;
        for (int i = 0; i < count; ++i) {
            muweight[i] = round(muweight[i] * n) / n;
        }
        // After Quantization, print
        if (this->phase_ == TRAIN && L == APP::show_layer && APP::step_ % APP::show_interval == 0) {
            Print(L, 'f');
        }
    }
    // -------------------------------------------
    
    
    
    #ifdef ShowTimingLog
    cout << " after prune, before GEMM: " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
    #endif

    
  /// ------------------------------------------------------
  
    const Dtype* weight = this->blobs_[0]->gpu_data();
    for (int i = 0; i < bottom.size(); ++i) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        Dtype* top_data = top[i]->mutable_gpu_data();
        for (int n = 0; n < this->num_; ++n) {
            this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
            top_data + n * this->top_dim_);
            if (this->bias_term_) {
                const Dtype* bias = this->blobs_[1]->gpu_data();
                this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
            }
        }
    }
    
    #ifdef ShowTimingLog
    cout << "  after GEMM: " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
    #endif

    /// this->bottom_dim_: bottom feature map size, input
    /// this->top_dim_: top feature map size, output
    /// this->num_: batch size
    
    /// Print feature map to check --------
    /// If row 3 and 8 are pruned in previous layer, then channel 3 and 8 will be only biases in this layer's feature map.
    /**
    if (!APP::IN_TEST && L == 0) {
        cout << "bottom.size(): " << bottom.size() << endl;
        for (int i = 0; i < bottom.size(); ++i) {
            const Dtype* top_data = top[i]->cpu_data();
            const int channel = top[i]->shape()[1];
            const int width   = top[i]->shape()[2];
            const int height  = top[i]->shape()[3];
            cout << "channel: " << channel << " " << width << " " <<  height << endl;
            
            vector<Dtype> sum(channel, 0);
            for (int c = 0; c < channel; ++c) {
                for (int w = 0 ; w < width; ++w) {
                    for (int h = 0; h < height; ++h) {
                        sum[c] += fabs(top_data[0 + c * width * height + w * height + h]);
                    }
                }
            }
            for (int c = 0; c < channel; ++c) {
                cout << sum[c] << "  ";
            }
            cout << endl;
        }
    }
    */
    /// -----------------------------------
    
    
    
    /// Restore weights ----------------
    if (this->IF_restore) {
        /// cout << layer_name << ": restore weights! " << endl;
        this->blobs_[0]->mutable_cpu_data();
        /// this->blobs_[0]->gpu_data(); 
        /// Interesting! If the above line is added, something like "control" seems to transfer from cpu to gpu. 
        /// Then modifying cpu weights won't affect their gpu counterparts.
        for (int i = 0; i < count; ++i) {
            muweight[i] = this->weight_backup[i];
        }
        
        /**
        /// ========================
        /// Chech restore
        cout << "weights from cpu:" << endl;
        for (int i = 0; i < 20; ++i) {
            cout << muweight[i] << " ";
        }
        cout << endl;

        Dtype weight_cpu[count];
        const Dtype* weight_gpu = this->blobs_[0]->gpu_data();
        cout << "weights copied from gpu:" << endl;
        cudaMemcpy(weight_cpu, weight_gpu, sizeof(Dtype) * count, cudaMemcpyDeviceToHost);
        for (int i = 0; i < 20; ++i) {
            cout << weight_cpu[i] << " ";
        }
        cout << endl;
        /// ========================
        */
    }
    /// --------------------------------
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
  
    /// ADDED BY WANGHUAN ------------------------------------------
    Dtype* muweight_diff = this->blobs_[0]->mutable_cpu_diff();      
    const int count   = this->blobs_[0]->count();
    const int L = APP::layer_index[this->layer_param_.name()];
    
    #ifdef ShowTimingLog
    cout << this->layer_param_.name() << " backward start timing" << endl;
    clock_t t1 = clock();
    #endif
    
    /// Print and check
    if (L == APP::show_layer && APP::step_ % APP::show_interval == 0 && APP::inner_iter == 0) {
        Print(L, 'b');
    }

    if (APP::prune_method != "None" && APP::pruned_ratio[L] > 0) {
        for (int j = 0; j < count; ++j) { 
            muweight_diff[j] *= APP::masks[L][j]; 
        }
    }
    #ifdef ShowTimingLog
    cout << "  after update diff: " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
    #endif
/// ------------------------------------------------------------- 
  
  
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
