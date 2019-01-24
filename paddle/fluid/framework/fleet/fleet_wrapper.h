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

#pragma once

#include <memory>
#ifdef PADDLE_WITH_PSLIB
#include <pslib.h>
#endif
#include <string>
#include <vector>
#include "paddle/fluid/platform/macros.h"  // for DISABLE_COPY_AND_ASSIGN

namespace paddle {
namespace framework {

// A wrapper class for pslib.h, this class follows Singleton pattern
// i.e. only initialized once in the current process
// Example:
//    std::shared_ptr<FleetWrapper> fleet_ptr =
//         FleetWrapper::GetInstance();
//    string dist_desc;
//    fleet_ptr->InitServer(dist_desc, 0);
// interface design principles:
// Pull
//   Sync: PullSparseVarsSync
//   Async: PullSparseVarsAsync(not implemented currently)
// Push
//   Sync: PushSparseVarsSync
//   Async: PushSparseVarsAsync
// Push dense variables to server in Async mode
// Param<in>: scope, table_id, var_names
// Param<out>: push_sparse_status

class FleetWrapper {
 public:
  FleetWrapper() {}
  virtual ~FleetWrapper() {}

  // Pull sparse variables from server in Sync mode
  // Param<in>: scope, table_id, var_names, fea_keys
  // Param<out>: fea_values
  void PullSparseVarsSync(
          const Scope& scope,
          const uint64_t table_id,
          const vector<std::string>& var_names,
          const std::vector<uint64_t>& fea_keys,
          std::vector<std::vector<float>>* fea_values);

  void PullDenseVarsSync(
          const Scope& scope,
          const uint64_t table_id,
          const vector<std::string>& var_names);

  void PullDenseVarsAsync(
          const Scope& scope,
          const uint64_t table_id,
          const vector<std::string>& var_names,
          std::vector<::std::future<int32_t>>* pull_dense_status>);

  // Push dense variables to server in async mode
  // Param<in>: scope, table_id, var_names,
  // Param<out>: push_sparse_status
  void PushDenseVarsAsync(
          const Scope& scope,
          const uint64_t table_id,
          const vector<std::string>& var_names,
          std::vector<::std::future<int32_t>>* push_sparse_status);

  // Push sparse variables with labels to server in Async mode
  // This is specially designed for click/show stats in server
  // Param<in>: scope, table_id, var_grad_names,
  //            fea_keys, fea_labels, sparse_grad_names
  // Param<out>: push_values, push_sparse_status
  void PushSparseVarsWithLabelAsync(
          const Scope& scope,
          const uint64_t table_id,
          const vector<uint64_t>& fea_keys,
          const vector<int64_t>& fea_labels,
          const vector<std::string>& sparse_grad_names,
          std::vector<std::vector<float>>* push_values,
          std::vector<::std::future<int32_t>>* push_sparse_status);

  // Push sparse variables to server in Async mode
  // Param<In>: scope, table_id, fea_keys, sparse_grad_names
  // Param<Out>: push_values, push_sparse_status
  void PushSparseVarsAsync(
          const Scope& scope,
          const uint64_t table_id,
          const vector<uint64_t>& fea_keys,
          const vector<std::string>& sparse_grad_names,
          std::vector<std::vector<float>>* push_values,
          std::vector<::std::future<int32_t>>* push_sparse_status);

  void InitServer(const std::string& dist_desc, int index);
  void InitWorker(const std::string& dist_desc,
                  const std::vector<uint64_t>& host_sign_list, int node_num,
                  int index);
  void StopServer();
  uint64_t RunServer();
  void GatherServers(const std::vector<uint64_t>& host_sign_list, int node_num);

  static std::shared_ptr<FleetWrapper> s_instance_ = NULL;
  static std::shared_ptr<FleetWrapper> GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::framework::FleetWrapper());
#ifdef PADDLE_WITH_PSLIB    
      _pslib_ptr = std::shared_ptr<paddle::distributed::PSlib>(
                               new paddle::distributed::PSlib());
#endif
    }
    return s_instance_;
  }

#ifdef PADDLE_WITH_PSLIB
  std::shared_ptr<paddle::distributed::PSlib> pslib_ptr_;
#endif

 protected:
  bool is_initialized_;
  DISABLE_COPY_AND_ASSIGN(FleetWrapper);
};

}  // end namespace framework
}  // end namespace paddle
