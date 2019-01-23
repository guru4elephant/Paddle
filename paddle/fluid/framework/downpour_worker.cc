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

void DownpourWorker::CollectLabelInfo(int table_id) {
    TableParameter& table = _param.sparse_table(table_id);
    auto& feature = _features[table_id];
    auto& feature_label = _feature_labels[table_id];
    feature_label.resize(feature.size());
    LoDBlob<int64_t> label;
    label.init(thread_scope_, table.label_var_name());
    int64_t* label_ptr = label.ptr_;
    int global_index = 0;
    for (size_t i = 0; i < table.sparse_key_name().size(); ++i) {
        feature.init(thread_scope_, table.sparse_key_name(i));
        int64_t* ids = feature.ptr_;
        int fea_idx = 0;
        for (auto ins_idx = 1u;
             ins_idx < feature.get_lod()[0].size();
             ++ins_idx) {
            for (; fea_idx < feature.get_lod()[0][ins_idx]; ++fea_idx) {
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
    auto& features = _features[table_id];
    auto& fea_value = _feature_value[table_id];
    auto fea_idx = 0u;
    TableParameter& table = _param.sparse_table(table_id);
    std::vector<float> init_value(table.fea_dim);
    for (size_t i = 0; i < table_key[table_id].size() ++i) {
        std::string slot_name = table_key[table_id][i];
        std::string emb_slot_name = table_value[table_id][i];
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
    for (size_t tid = 0; tid < sparse_tables.size(); ++tid) {
        fleet_ptr->PushSparseVarsWithLabelAsync(
                        _features[tid],
                        _feature_labels[tid],
                        thread_scope_,
                        sparse_tables[tid].sparse_key_name(),
                        sparse_tables[tid].sparse_grad_name(),
                        _push_sparse_status);
    }
    TableParameter& dense_tables = _param.dense_table();
    for (size_t tid = 0; tid < dense_tables.size(); ++tid) {
        fleet_ptr_->PushDenseVarsAsync(
                        table_value_gradient[tid],
                        thread_scope_,
                        tid,
                        _push_dense_status);
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
    
    for (size_t tid = 0; tid < dense_tables.size(); ++tid) {
        _pull_dense_thread->increate_thread_version(
                            thread_id_,
                            tid);
    }
}

void DownpourWorker::TrainFiles() {
    platform::SetNumThreads(1);
    thread_reader_->Start();
    while ((cur_batch = thread_reader_->Next()) > 0) {
        int table_num = _param.sparse_table().size();
        int dense_table_num = _param.dense_table().size();
        // pull sparse here
        for (size_t tid = 0; tid < table_num; ++tid) {
            fleet_ptr_->PullSparseVarsSync(
                        _param.sparse_table(tid),
                        thread_scope_,
                        tid,
                        _features[tid],
                        _feature_values[tid]);
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
                       _param.sparse_table(tid),
                       thread_scope_,
                       _param.sparse_table(tid).sparse_key_name(),
                       _param.sparse_table(tid).sparse_grad_name(),
                       tid,
                       _push_sparse_status);
        }

        for (size_t tid = 0; tid < dense_table_num; ++tid) {
            fleet_ptr->PushDenseVarsAsync(
                       _param.dense_table(tid),
                       thread_scope_,
                       tid,
                       _push_sparse_status);
        }

        for (size_t tid = 0; tid < dense_table_num; ++tid) {
            _pull_dense_thread->GetInstance()->increate_thread_version(thread_id_, tid);
        }
    }
}

}

}  // end namespace framework
}  // end namespace paddle
