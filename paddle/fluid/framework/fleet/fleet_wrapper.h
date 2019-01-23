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
class FleetWrapper {
 public:
  FleetWrapper() {}
  virtual ~FleetWrapper() {}

  void PushDenseVarsAsync(
          vector<std::string>& var_names,
          const Scope& scope, uint64_t table_id,
          std::vector<::std::future<int32_t>>* push_sparse_status);

  void PullSparseVarsSync(
          vector<std::string>& var_names,
          const Scope& scope, uint64_t table_id,
          std::vector<uint64_t>* fea_keys,
          std::vector<std::vector<float>>* fea_values);

  void PushSparseVarsWithLabelAsync(
          const vector<uint64_t>& fea_keys,
          const vector<int64_t>& fea_labels,
          const Scope& scope,
          vector<std::string>& value_var_names,
          std::vector<std::vector<float>>* push_values,
          uint64_t table_id,
          std::vector<::std::future<int32_t>>* push_sparse_status);

  void PushSparseVarsAsync(
          const vector<uint64_t>& fea_keys,
          const Scope& scope,
          vector<std::string>& value_var_names,
          std::vector<std::vector<float>>* push_values,
          uint64_t table_id,
          std::vector<::std::future<int32_t>>* push_sparse_status);

  std::future<uint32_t> WorkerPullSparse();
  std::future<uint32_t> WorkerPushSparse();
  std::future<uint32_t> WorkerPullDense();
  std::future<uint32_t> WorkerPushDense();
  void InitServer(const std::string& dist_desc, int index);
  void InitWorker(const std::string& dist_desc,
                  const std::vector<uint64_t>& host_sign_list, int node_num,
                  int index);
  void StopServer();
  uint64_t RunServer();
  void GatherServers(const std::vector<uint64_t>& host_sign_list, int node_num);

  static std::shared_ptr<FleetWrapper> _s_instance;
  static std::shared_ptr<FleetWrapper> GetInstance() {
    if (NULL == _s_instance) {
      _s_instance.reset(new paddle::framework::FleetWrapper());
    }
    return _s_instance;
  }

#ifdef PADDLE_WITH_PSLIB
  std::shared_ptr<paddle::distributed::PSlib> _pslib_ptr;
#endif

 protected:
  bool is_initialized_;
  DISABLE_COPY_AND_ASSIGN(FleetWrapper);
};

}  // end namespace framework
}  // end namespace paddle
