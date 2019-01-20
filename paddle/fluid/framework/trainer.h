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
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/trainer_desc.pb.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/operators/reader/blocking_queue.h"

namespace paddle {
namespace framework {

class TrainerBase {
 public:
  TrainerBase() {}
  virtual ~TrainerBase() {}
  // model memory are hosted in root_scope
  void SetScope(Scope* root_scope);
  // device worker and data feed will be created
  virtual void Initialize(const TrainerDesc& trainer_desc);
  // whether we need to run debug mode
  void SetDebug(const bool debug) { debug_ = debug; }
  // create execution resources for training
  // mainly call CreateDeviceResouce of DeviceWorker
  virtual void InitTrainerEnv(const ProgramDesc& main_program,
                              const platform::Place& place) = 0;
  // create other execution resources for training
  // for example, prepare an helper thread in this function
  virtual void InitOtherEnv(const ProgramDesc& main_program) = 0;
  // run computation logic in device workers here
  virtual void Run() = 0;
  // finalize all resources created in Initialize()
  virtual void Finalize() = 0;

 protected:
  Scope* root_scope_;
  bool debug_;
};

// general trainer for async execution
// local trainer and distributed trainer are supported
// depends on the assigned device_worker
class MultiTrainer : public TrainerBase {
 public:
  MultiTrainer() {}
  virtual ~MultiTrainer() {}
  virtual void Initialize(const TrainerDesc& trainer_desc);
  virtual void InitTrainerEnv(const ProgramDesc& main_program,
                              const platform::Place& place);
  virtual void InitOtherEnv(const ProgramDesc& main_program) {}
  virtual void Run();
  virtual void Finalize();

 protected:
  int thread_num_;
  std::vector<std::thread> threads_;
  std::vector<std::shared_ptr<DataFeed>> readers_;
  std::vector<std::shared_ptr<DeviceWorker>> workers_;
};

/*
class DistributedMultiTrainer : public MultiTrainer {
 public:
  DistributedMultiTrainer() {}
  virtual ~DistributedMultiTrainer() {}
  virtual void InitOtherEnv();

 protected:
  std::shared_ptr<PullDenseWorker> pull_dense_thread_;
}
*/

}  // namespace framework
}  // namespace paddle
