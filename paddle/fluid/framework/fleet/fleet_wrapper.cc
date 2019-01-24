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

#include "paddle/fluid/framework/fleet/fleet_wrapper.h"

namespace paddle {
namespace framework {

static const uint32_t MAX_FEASIGN_NUM = 1024 * 100 * 100;

void FleetWrapper::InitServer(const std::string& dist_desc, int index) {
  pslib_ptr_->init_server(dist_desc, index);
}

void FleetWrapper::InitWorker(const std::string& dist_desc,
                              const std::vector<uint64_t>& host_sign_list,
                              int node_num, int index) {
  pslib_ptr_->init_worker(
      dist_desc, const_cast<uint64_t*>(host_sign_list.data()), node_num, index);
}

void FleetWrapper::StopServer() { pslib_ptr_->stop_server(); }

uint64_t FleetWrapper::RunServer() { return pslib_ptr_->run_server(); }

void FleetWrapper::GatherServers(const std::vector<uint64_t>& host_sign_list,
                                 int node_num) {
  pslib_ptr_->gather_servers(const_cast<uint64_t*>(host_sign_list.data()),
                             node_num);
}



void FleetWrapper::PullSparseVarsSync(
                   vector<std::string>& var_names,
                   const Scope& scope, uint64_t table_id,
                   std::vector<uint64_t>* fea_keys,
                   std::vector<std::vector<float>>* fea_values,
                   int fea_value_dim) {
    std::vector<::std::future<int32_t>> pull_sparse_status;
    pull_sparse_status.resize(0);
    fea_keys.clear();
    fea_keys.resize(0);
    features.reserve(MAX_FEASIGN_NUM);
    for (auto name : var_names) {
        Variable* var = scope->FindVar(name);
        LoDTensor* tensor = var->GetMutable<LoDTensor>();
        int64_t * ids = tensor->data<int64_t>();
        int len = tensor->numel();
        for (auto i = 0u; i < len; ++i) {
            if (ids[i] == 0u) {
                continue;
            }
            fea_keys.push_back(static_cast<uint64_t>(ids[i]));
        }
        fea_values.resize(fea_keys.size() + 1);
        for (auto& t : fea_values) {
            t.resize(fea_value_dim);
        }
        std::vector<float *> pull_result_ptr;
        for (auto& t : fea_values) {
            pull_result_ptr.push_back(t.data());
        }
        auto status = pslib_ptr_->_worker_ptr->pull_sparse(
            pull_result_ptr.data(), table_id,
            fea_keys.data(), fea_keys.size());
        pull_sparse_status.push_back(std::move(status));
    }
    for (auto& t : pull_sparse_status) {
        t.wait();
        auto status = t.get();
        if (status != 0) {
            LOG(ERROR) << "fleet pull sparse failed, status[" << status << "]";
            exit(-1);
        }
    }
}

void FleetWrapper::PullDenseVarsAsync(
                   const Scope& scope,
                   const uint64_t tid,
                   const vector<std::string>& var_names,
                   std::vector<::std::future<int32_t>>* pull_dense_status>) {
    std::vector<paddle::ps::Region> regions;
    regions.reserve(var_names.size());
    for (auto& t : var_names) {
        Variable* var = scope->FindVar(t);
        LoDTensor* tensor = var->GetMutable<LoDTensor>();
        float* w = tensor->data<float>();
        paddle::ps::Region reg(w, tensor->numel());
        regions.emplace_back(std::move(reg));
    }
    auto& status = pslib_ptr_->worker_ptr->pull_dense(
                    regions.data(), regions.size(), tid);
    pull_dense_status.push_back(status);
}

void FleetWrapper::PullDenseVarsSync(
                   const Scope& scope,
                   const uint64_t tid,
                   const vector<std::string>& var_names) {
    std::vector<paddle::ps::Region> regions;
    regions.reserve(var_names.size());
    for (auto& t : var_names) {
        Variable* var = scope->FindVar(t);
        LoDTensor* tensor = var->GetMutable<LoDTensor>();
        float* w = tensor->data<float>();
        paddle::ps::Region reg(w, tensor->numel());
        regions.emplace_back(std::move(reg));
    }
    _pull_dense_status.push_back(
        pslib_ptr_->worker_ptr->pull_dense(regions.data(), regions.size(), tid));
    for (auto& t : _pull_dense_status) {
        t.wait();
        auto status = t.get();
        if (status != 0) {
            LOG(WARNING) << "pull dense failed times:" << ++_pull_dense_fail_times;
        }
    }

    if (_pull_dense_fail_times > 20) {
        LOG(FATAL) << "pull dense failed times more than 20 times";
        exit(-1);
    }
    _pull_dense_status.resize(0);
}

void FleetWrapper::PushDenseVarsAsync(
                   vector<std::string>& var_names,
                   const Scope& scope, int table_id,
                   std::vector<::std::future<int32_t>>* push_sparse_status) {
    std::vector<paddle::ps::Region> regions;
    for (auto& t : var_names) {
        Variable* var = scope.FindVar(t);
        LoDTensor* tensor = var->GetMutable<LoDTensor>();
        int count = tensor->numel();
        float* g = tensor->data<float>();
        paddle::ps::Region reg(g, count);
        regions.emplace_back(std::move(reg));
    }
    auto status = pslib_ptr_->worker_ptr->push_dense(
            regions.data(), regions.size(), table_id);
    push_sparse_status.push_back(status);
}

}  // end namespace framework
}  // end namespace paddle
