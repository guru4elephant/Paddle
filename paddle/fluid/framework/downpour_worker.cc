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
#include "paddle/framework/device_worker_base.h"

void DownpourWorker::CreateDeviceResource(
    const ProgramDesc& main_prog, const paddle::platform::Place& place) {
  CreateThreadScope(program);
  CreateThreadOperators(program);
  SetPlace(place);
}

void DownpourWorker::TrainFiles() {
  platform::SetNumThreads(1);
  // binding cpu here?

  thread_reader_->Start();
  int cur_batch;
  int batch_cnt = 0;
  while ((cur_batch = thread_reader_->Next()) > 0) {
    TrainOneNetwork();
    ++batch_cnt;
    thread_scope_->DropKids();
  }
}

void DownpourWorker::TrainFilesWithProfiler() {}
