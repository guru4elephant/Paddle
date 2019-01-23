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

#include <fstream>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/device_worker.pb.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/operators/reader/blocking_queue.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/timer.h"

namespace paddle {
namespace framework {

class PullDenseWorker {
 public:
  PullDenseWorker() {}
  virtual ~PullDenseWorker() {}
  virtual void Initialize(const PullDenseWorkerParameter& param);
  void Start();
  void Stop();
  void IncreaseThreadVersion(int thread_id, uint64_t table_id);
  void ResetThreadVersion(uint64_t table_id);
  void WaitAll();
 private:
  void Run();
  bool CheckUpdateParam(uint64_t table_id);
  
 private:
  std::shared_ptr<paddle::framework::FleetWrapper> fleet_ptr_;
  PullDenseWorkerParameter _param;
  Scope* _root_scope;
  bool _running;

  std::map<uint64_t, uint64_t> _last_versions;
  std::map<uint64_t, uint64_t> _current_version;
  std::mutex _mutex_for_version;
  std::map<uint64_t, std::vector<uint64_t>> _training_versions;
  std::map<uint64_t, std::vector<std::string>> _dense_variable_name;
  
  std::thread _t;

  std::vector<::std::future<int32_t>> _pull_dense_status;
  uint32_t _pull_dense_fail_times = 0;
  std::vector<float> _base_norm_param;
  std::vector<float> _mean;
  std::vector<float> _scale;
  float _squared_sum_epsilon = 1e-4;
  std::mutex _mutex_for_mean_scale;
  float _total_batch_num = 0;
};

// should incorporate different type of device
class DeviceWorker {
 public:
  DeviceWorker() {}
  virtual ~DeviceWorker() {}
  virtual void Initialize(const DeviceWorkerDesc& desc) = 0;
  virtual void SetDeviceIndex(int tid) = 0;
  virtual void TrainFiles() = 0;
  virtual void TrainFilesWithProfiler() = 0;
  virtual void CreateDeviceResource(const ProgramDesc& main_prog) = 0;
  // will make this zero copy in the future
  virtual void BindingDataFeedMemory() = 0;
  virtual void SetRootScope(Scope* root_scope);
  virtual void SetDataFeed(const std::shared_ptr<DataFeed>& data_feed);
  virtual void SetPlace(const paddle::platform::Place& place) {
    place_ = place;
  }

 protected:
  Scope* root_scope_;
  paddle::platform::Place place_;
  std::shared_ptr<DataFeed> device_reader_;
};

class CPUWorkerBase : public DeviceWorker {
 public:
  CPUWorkerBase() {}
  virtual ~CPUWorkerBase() {}
  virtual void SetDeviceIndex(int tid) { thread_id_ = tid; }
  virtual void TrainFiles() = 0;
  virtual void TrainFilesWithProfiler() {}
  virtual void CreateDeviceResource(const ProgramDesc& main_prog) {}

 protected:
  int thread_id_;
};

class HogwildWorker : public CPUWorkerBase {
 public:
  HogwildWorker() {}
  virtual ~HogwildWorker() {}
  virtual void Initialize(const DeviceWorkerDesc& desc) {}
  virtual void TrainFiles();
  virtual void TrainFilesWithProfiler();
  virtual void CreateDeviceResource(const ProgramDesc& main_prog);
  virtual void BindingDataFeedMemory();

 protected:
  void CreateThreadOperators(const ProgramDesc& program);
  void CreateThreadScope(const ProgramDesc& program);
  std::shared_ptr<DataFeed> thread_reader_;
  std::vector<std::string> op_names_;
  std::vector<OperatorBase*> ops_;
  Scope* thread_scope_;
  std::vector<std::string> fetch_var_names_;
  std::vector<std::vector<float>> fetch_values_;
  platform::Place place_;
};


class DownpourWorker : public HogwildWorker {
 public:
  DownpourWorker() {}
  virtual ~DownpourWorker() {}
  virtual void TrainFiles();
  virtual void TrainFilesWithProfiler();
  virtual void CreateDeviceResource(const ProgramDesc& main_prog,
                                    const paddle::platform::Place& place);

 protected:
  std::shared_ptr<paddle::framework::FleetWrapper> fleet_ptr_;
  void PushSparse(int table_id);
  void PushDense(int table_id);
  void PullSparse();
  void FillSparseValue(int table_id);
  void PushGradients();
  void CollectLabelInfo(int table_id);

 private:
  DownpourWorkerParameter _param;
  struct ParamConfig {
    int slot_dim;
    int fea_dim;
    int32_t tmp_push_dense_wait_times;
    int32_t tmp_push_sparse_wait_times;
    std::vector<std::string> skip_op;
    std::map<uint64_t, std::vector<std::string>> dense_variable_name;
    std::vector<int> dense_table_id;
    std::vector<uint32_t> dense_table_size;
    std::vector<int> sparse_table_id;
    std::map<uint64_t, std::vector<std::string>> slot_input_vec;
    std::map<uint64_t, std::vector<std::string>> gradient_var;
    std::map<std::string, uint64_t> slot_alias_to_table;
  };

  std::map<uint64_t, std::vector<std::string>> _table_key;
  std::map<uint64_t, std::vector<std::string>> _table_value;
  std::map<uint64_t, std::vector<std::string>> _table_value_gradient;

  // feasign
  std::map<uint64_t, std::vector<uint64_t>> _features;
  // feasign stats
  std::map<uint64_t, std::vector<float>> _feature_labels;
  // feasign embedding
  std::map<uint64_t, std::vector<std::vector<float>>> _feature_values;
  // feasign embedding gradient
  std::map<uint64_t, std::vector<std::vector<float>>> _feature_push_values;
  // fleet handler
  std::shared_ptr<paddle::framework::FleetWrapper> fleet_ptr_;

  std::shared_ptr<DensePullThread> _pull_dense_thread;
  std::vector<::std::future<int32_t>> _push_sparse_status;
  std::vector<::std::future<int32_t>> _push_dense_status;
};

}  // namespace framework
}  // namespace paddle
