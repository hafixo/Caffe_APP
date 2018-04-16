#include <vector>
#include <ctime>
#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/adaptive_probabilistic_pruning.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
        
    // -----------------------------------------------
    // Added by WANGHUAN for pruning
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const string layer_name = this->layer_param_.name();
    const string mthd = APP::prune_method;
    char* coremthd = new char[strlen(APP::prune_coremthd.c_str()) + 1];
    strcpy(coremthd, APP::prune_coremthd.c_str());
    const string coremthd_ = strtok(coremthd, "-");
    const int L = APP::layer_index[layer_name];
    
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
                if (APP::prune_unit == "Col" && L != APP::conv_layer_cnt + APP::fc_layer_cnt -1) {
                    if (APP::step_-1 - APP::iter_prune_finished[L+1] <= 1) {
                        //UpdateNumPrunedRow();
                    }
                } else if (APP::prune_unit == "Row" && mthd != "TP_Row" && APP::pruned_rows.size()) {
                    UpdateNumPrunedCol();
                }
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

                    // print to log
                    char rlayer[10];
                    char rrow[10];
                    char rcol[10];
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
        
        // Print, before masked
        if (L == APP::show_layer + APP::conv_layer_cnt && APP::step_ % APP::show_interval == 0) {
            Print(L, 'f');
        }
        
        // Update masks
        if (IF_prune && APP::iter_prune_finished[L] == INT_MAX) {
            if (APP::prune_coremthd.substr(0,3) == "Reg") {
                if (APP::step_ % 100 == 0) {  PruneMinimals(); }
                #ifdef ShowTimingLog
                cout << "  after PruneMinimals: " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
                #endif
            } else if (APP::prune_coremthd.substr(0,2)== "PP" && APP::prune_unit == "Weight") {
                ProbPruneWeight(APP::prune_interval);
            }
            UpdatePrunedRatio();
            if (L == APP::conv_layer_cnt + APP::fc_layer_cnt - 1) { // To avoid the first conv from updating col
                APP::pruned_rows.clear();
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
                 << "  pruned_ratio: " << APP::pruned_ratio[L] 
                 << "  prune_ratio: " << APP::prune_ratio[L] 
		 << "  num_negative: " << cnt_negative << "(" << cnt_negative*1.0/count << ")" << endl;
        }
    } else if (this->phase_ == TEST && IF_prune && APP::iter_prune_finished[L] == INT_MAX && coremthd_ == "PP") {
        if (APP::prune_unit == "Weight") {
            const int num_batch = 10;
            Dtype rands[count/num_batch];
            for (int i = 0; i < count; ++i) {
                if (i % (count/num_batch) == 0) {
                    caffe_rng_uniform(count/num_batch, (Dtype)0, (Dtype)1, rands);
                }
                APP::masks[L][i] = rands[i%(count/num_batch)] < APP::history_prob[L][i] ? 1 : 0;
                this->weight_backup[i] = muweight[i];
                muweight[i] *= APP::masks[L][i];
            }
            this->IF_restore = true;
        }
    }
    #ifdef ShowTimingLog
    cout << "  before GEMM: " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
    #endif
  // ------------------------------------------------
  
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
  #ifdef ShowTimingLog
  cout << "  after GEMM, end of foward: " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
  #endif
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    }
    
    // -------------------------------------------------
    const int L = APP::layer_index[this->layer_param_.name()];
    // Print
    if (L == APP::show_layer + APP::conv_layer_cnt && APP::step_ % APP::show_interval == 0 && APP::inner_iter == 0) {
        Print(L, 'b');
    }
        
    if (APP::prune_method != "None" && APP::pruned_ratio[L] > 0) {
        const int count = this->blobs_[0]->count();
        Dtype* muweight_diff = this->blobs_[0]->mutable_cpu_diff();
        for (int i = 0; i < count; ++i) {
            muweight_diff[i] *= APP::masks[L][i];
        }
    }
    // -------------------------------------------------
    
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
         (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
  }
  
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe
