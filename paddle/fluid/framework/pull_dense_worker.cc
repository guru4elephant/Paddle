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

namespace paddle {
namespace framework {

void PullDenseWorker::Initialize(const PullDenseWorkerParameter& param) {
    running_ = false;
    param_ = param;
    threshold_ = param.threshold;
    thread_num_ = param.training_thread_num;
    sleep_time_ms_ = param.sleep_time_ms;
    for (size_t i = 0; i < param.dense_table_size(); ++i) {
        // setup dense variables for each table
        int var_num = param.table().dense_value_name_size();
        dense_variable_name_[param.table_id()].resize(var_num);
        for (int j = 0; j < var_num; ++j) {
            dense_variable_name_[param.table_id][j] =
                param.table().dense_value_name(j);
        }
        // setup training version for each table
        training_versions_[param.table_id()].resize(thread_num_, 0);
        last_versions_[param.table_id()] = 0;
        current_version_[param.table_id()] = 0;
    }
}

void PullDenseWorker::Wait(
     const std::vector<::std::future<int32_t>>* status_vec) {
    for (auto& t : status_vec) {
        t.wait();
        auto status = t.get();
        if (status != 0) {
            LOG(WARNING) << "Current Pull Dense Thread Failed Times" <<
                ++pull_dense_fail_times_;
        }
    }

    if (pull_dense_fail_times_ > MAX_FAIL_NUM) {
        LOG(FATAL) << "Pull Dense Failed Times More Than " << 
            MAX_FAIL_NUM << " Times";
        exit(-1);
    }
}

void PullDenseWorker::Stop() {
    if (running_) {
        running_ = false;
        t_.join();
    }
}

int PullDenseWorker::Start() {
    running_ = true;
    t_ = std::thread(&DensePullThread::Run, this);
    return 0;
}

void PullDenseWorker::Run() {
    while (_running) {
        pull_dense_status_.resize(0);
        for (auto& t : dense_variable_name_) {
            
        }
        if (pull_dense_status_.size() != 0) {
            for (auto)
        }
        
        usleep(sleep_time_ms_);
    }
}

void PullDenseWorker::IncreateThreadVersion(
                      int thread_id, uint64_t table_id) {
    std::lock_guard<std::mutex> lock(mutex_for_version_);
    training_versions_[table_id][thread_id]++;
}


bool PullDenseWorker::CheckUpdateParam(uint64_t table_id) {
    std::lock_guard<std::mutex> lock(mutex_for_version_);
    auto& version = training_versions_[table_id];
    current_version_[table_id] = *(std::min_element(version.begin(),
                                                    version.end()));
    if (current_version_[table_id] - last_versions_[table_id] < threshold_) {
        return false;
    }
    return true;
}

void PullDenseWorker::ResetThreadVersion(uint64_t table_id) {
    std::lock_guard<std::mutex> lock(mutex_for_version_);
    last_versions_[table_id] = current_version_[table_id];
}


}  // namespace framework
}  // namespace paddle
