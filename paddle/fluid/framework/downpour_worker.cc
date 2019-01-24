/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/platform/cpu_helper.h"

namespace paddle {
namespace framework {

void DownpourWorker::Initilize(const DeviceWorkerDesc& desc) {
  param_ = desc.downpour_param();

  for (size_t i = 0; i < param_.sparse_table_size(); ++i) {
    int64_t table_id = param_.sparse_table(i).table_id;
    TableParameter& table = param_.sparse_table(i);
    sparse_key_name_[table_id].resize(table.sparse_key_name_size());
    for (size_t j = 0; j < table.sparse_key_name_size(); ++j) {
      sparse_key_name_[table_id][j] = table.sparse_key_name(j);
    }
    sparse_value_name_[table_id].resize(table.sparse_value_name_size());
    for (size_t j = 0; j < table.sparse_value_name_size(); ++j) {
      sparse_value_name_[table_id][j] = table.sparse_value_name(j);
    }
    sparse_grad_name_[table_id].resize(table.sparse_grad_name_size());
    for (size_t j = 0; j < table.sparse_grad_name_size(); ++j) {
      sparse_grad_name_[table_id][j] = table.sparse_grad_name(j);
    }
  }

  for (size_t i = 0; i < param_.dense_table_size(); ++i) {
    int64_t table_id = param_.dense_table(i).table_id;
    TableParameter& table = param_.dense_table(i);
    dense_value_name_[table_id].resize(table.dense_value_name_size());
    for (size_t j = 0; j < table.dense_value_name_size(); ++j) {
      dense_value_name_[table_id][j] = table.dense_value_name(j);
    }
    dense_grad_name_[table_id].resize(table.dense_grad_name_size());
    for (size_t j = 0; j < table.dense_grad_name_size(); ++j) {
      dense_grad_name_[table_id][j] = table.dense_grad_name(j);
    }
  }

  skip_ops_.resize(param_.skip_ops_size());
  for (size_t i = 0; i < param_.skip_ops_size(); ++i) {
    skip_ops_[i] = param_.skip_ops(i);
  }
}

void DownpourWorker::CollectLabelInfo(int table_id) {
  //TableParameter& table = _param.sparse_table(table_id);
  auto& feature = features_[table_id];
  auto& feature_label = feature_labels_[table_id];
  feature_label.resize(feature.size());
  Variable* var = thread_scope_->FindVar(label_var_name_);
  LoDTensor* tensor = var->GetMutable<LoDTensor>();
  int64_t* label_ptr = tensor->data<int64_t>();
  int global_index = 0;
  for (size_t i = 0; i < sparse_key_names_[table_id].size(); ++i) {
    Variable* fea_var =
        thread_scope_->FindVar(sparse_key_names_[table_id][i]);
    LoDTensor* tensor = fea_var->GetMutable<LoDTensor>();
    int64_t* ids = tensor->data<int64_t>();
    int fea_idx = 0;
    for (auto ins_idx = 1u;
      ins_idx < tensor->lod()[0].size();
         ++ins_idx) {
      for (; fea_idx < tensor->lod()[0][ins_idx]; ++fea_idx) {
        // should be skipped feasign defined in protobuf
        if (ids[fea_idx] == 0u) {
          continue;
        }
        feature_label[global_index++] = ids[fea_idx];
      }
    }
  }
}

void DownpourWorker::FillSparseValue(int table_id) {
  auto& features = features_[table_id];
  auto& fea_value = feature_values_[table_id];
  auto fea_idx = 0u;
  TableParameter& table = _param.sparse_table(table_id);
  std::vector<float> init_value(table.fea_dim);
  for (size_t i = 0; i < sparse_key_names_[table_id].size() ++i) {
    std::string slot_name = sparse_key_names_[table_id][i];
    std::string emb_slot_name = sparse_value_names_[table_id][i];
    Variable* var = thread_scope_->FindVar(slot_name);
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    int64_t* ids = tensor->data<int64_t>();
    int len = tensor->numel();
    Variable* var_emb = thread_scope_->FindVar(emb_slot_name);
    LoDTensor* tensor_emb = var_emb->GetMutable<LoDTensor>();
    float* ptr = 
        tensor_emb->mutable_data<float>(
            {len, table.slot_dim},
            platform::CPUPlace());
    memset(ptr, 0, sizeof(float) * len * table.slot_dim);
    auto& tensor_lod = tensor->lod()[0];
    LoD data_lod{tensor_lod};
    tensor_emb->set_lod(data_lod);
    for (auto index = 0u; index < len; ++index) {
      if (ids[index] == 0u) {
        memcpy(ptr + table.slot_dim * index,
               init_value.data() + 2,
               sizeof(float) * table.slot_dim);
        continue;
      }
      memcpy(ptr + table.slot_dim * index,
             fea_value[fea_idx].data() + 2,
             sizeof(float) * table.slot_dim);
      fea_idx++;
    }
  }
}

// checked
void DownpourWorker::PushGradients() {
  TableParameter& sparse_tables = _param.sparse_table();
  for (size_t tid = 0; tid < sparse_key_names_.size(); ++tid) {
    fleet_ptr->PushSparseVarsWithLabelAsync(
               features_[tid],
               feature_labels_[tid],
               thread_scope_,
               sparse_key_names_[tid],
               sparse_grad_names_[tid],
               push_sparse_status_);
  }

  for (size_t tid = 0; tid < dense_grad_names_.size(); ++tid) {
    fleet_ptr_->PushDenseVarsAsync(
                dense_grad_names_[tid],
                thread_scope_,
                tid,
                push_dense_status_);
  }
  /*
    int32_t tmp_push_dense_wait_times = -1;
    int32_t tmp_push_sparse_wait_times = -1;
    static uint32_t push_dense_wait_times =
    static_cast<uint32_t>(tmp_push_dense_wait_times);
    static uint32_t push_sparse_wait_times =
    static_cast<uint32_t>(tmp_push_sparse_wait_times);
    if (_push_dense_status.size() >= _param.push_dense_wait_times) {
    for (auto& t : _push_dense_status) {
    t.wait();
    }
    _push_dense_status.resize(0);
    }
  */
    
  for (size_t tid = 0; tid < dense_value_names_.size(); ++tid) {
    pull_dense_thread_->increate_thread_version(
                        thread_id_,
                        tid);
  }
}

void DownpourWorker::TrainFiles() {
  platform::SetNumThreads(1);
  thread_reader_->Start();
  while ((cur_batch = thread_reader_->Next()) > 0) {
    int table_num = sparse_value_names_.size();
    int dense_table_num = dense_value_names_.size();
    // pull sparse here
    for (size_t tid = 0; tid < table_num; ++tid) {
      fleet_ptr_->PullSparseVarsSync(
                  sparse_key_names_[tid],
                  thread_scope_,
                  tid,
                  features_[tid],
                  feature_values_[tid]);
      CollectLabelInfo(tid);
      FillSparseValue(tid);
    }
    
    // do computation here
    for (auto& op : ops_) {
      op->Run(*thread_scope_, place_);
    }
    
    // push gradients here
    for (size_t tid = 0; tid < table_num; ++tid) {
      fleet_ptr->PushSparseVarsWithLabelAsync(
                 features_[tid],
                 feature_labels_[tid],
                 thread_scope_,
                 sparse_keys_names_[tid],
                 sparse_grad_names_[tid],
                 tid,
                 push_sparse_status_);
    }
    
    for (size_t tid = 0; tid < dense_table_num; ++tid) {
      fleet_ptr->PushDenseVarsAsync(
                 dense_grad_names_[tid],
                 thread_scope_,
                 tid,
                 push_sparse_status_);
    }

    for (size_t tid = 0; tid < dense_table_num; ++tid) {
      pull_dense_thread_->GetInstance()->increate_thread_version(thread_id_, tid);
    }
  }
}

}

}  // end namespace framework
}  // end namespace paddle
