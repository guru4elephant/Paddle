/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

int PullDenseWorker::Start() {
  _running = true;
  _t = std::thread(&PullDenseWorker::Run, this);
  return 0;
}

void PullDenseWorker::Run() {
  while (_running) {
    _pull_dense_status.resize(0);
    for (auto& t : _dense_variable_name) {
      auto status = PullDense();
      _pull_dense_status.emplace_back(std::move(status));
      reset_thread_version(t.first);
    }
  }
  if (_pull_dense_status.size() != 0) {
    wait_all();
  }

  usleep(_sleep_time_ms * 1000);
}

void PullDenseWorker::Stop() {
  if (_running) {
    _running = false;
    _t.join();
  }
}

void PullDenseWorker::IncreateThreadVersion(int thread_id, uint64_t table_id) {
  std::lock_guard<std::mutex> lock(_mutex_for_version);
  _training_versions[table_id][thread_id]++;
}

void PullDenseWorker::ResetThreadVersion(uint64_t table_id) {
  std::lock_guard<std::mutex> lock(_mutex_for_version);
  _last_versions[table_id] = _current_version[table_id];
}

void PullDenseWorker::WaitAll() {
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

bool PullDenseWorker::CheckUpdateParam(uint64_t table_id) {
  std::lock_guard<std::mutex> lock(_mutex_for_version);
  auto& version = _training_versions[table_id];
  _current_version[table_id] =
      *(std::min_element(version.begin(), version.end()));
  if (_current_version[table_id] - _last_versions[table_id] < _threshold) {
    return false;
  }
  return true;
}

void PullDenseWorker::ResetThreadVersion(uint64_t table_id) {
  std::lock_guard<std::mutex> lock(_mutex_for_version);
  _last_versions[table_id] = _current_version[table_id];
}

std::future<int32_t> PullDenseWorker::PullDense(uint64_t table_id) {
  auto& regions = _regions[table_id];
  regions.clear();
  auto& variables = _dense_variable_name[table_id];
  regions.resize(variables.size());

  for (auto i = 0u; i < variables.size(); ++i) {
    auto& t = variables[i];
    Variable* var = _root_scope->FindVar(t);
    LoDTensor* tensor = var->GetMutable<LoDTensor>();

    float* w = tensor->data<float>();
    paddle::ps::Region reg(w, tensor->numel());
    regions[i] = std::move(reg);
  }
  return _pslib_ptr->worker_ptr->pull_dense(regions.data(), regions.size(),
                                            table_id);
}
