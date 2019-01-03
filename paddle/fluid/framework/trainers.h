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

#include "paddle/fluid/framework/data_feed.pb.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/operators/reader/blocking_queue.h"

namespace paddle {
namespace framework {

class TrainerBase {
 public:
  TrainerBase() {}
  virtual ~TrainerBase() {}
  // model memory are hosted in root_scope
  void SetScope(Scope* root_scope);
  void SetMainProgramDesc(const ProgramDesc& program);
  void Initialize(const TrainerDesc& trainer_desc);
  void SetDebug(const bool debug) { debug_ = debug; }
  virtual void InitTrainerEnv() = 0;
  virtual void InitOtherEnv() = 0;
  virtual void Run() = 0;
  virtual void Finalize() = 0;

 protected:
  ProgramDesc main_program_;
  TrainerDesc trainer_desc_;
  Place place_;
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
  virtual void InitTrainerEnv();
  virtual void InitOtherEnv();
  virtual void Run();
  virtual void Finalize();

 protected:
  virtual void InitReaders();
  virtual void InitDeviceWorkers();
  int thread_num_;
  platform::Place place_;
  std::vector<std::thread> threads_;
  std::vector<std::shared_ptr<DataFeed>> readers_;
  std::vector<std::shared_ptr<DeviceWorkerBase>> workers_;
};

class DistributedMultiTrainer : public MultiTrainer {
 public:
  DistributedMultiTrainer() {}
  virtual ~DistributedMultiTrainer() {}
  virtual void Initialize(const TrainerDesc& trainer_desc);
  virtual void InitTrainerEnv();
  virtual void InitOtherEnv();
  virtual void Run();
  virtual void Finalize();
}

}  // namespace framework
}  // namespace paddle
