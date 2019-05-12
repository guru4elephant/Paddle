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

#pragma once

#include <algorithm>
#include <condition_variable>  // NOLINT
#include <deque>
#include <mutex>  // NOLINT
#include <utility>
#include <vector>

namespace paddle {
namespace framework {

template <typename T>
class BlockingQueue {
 public:
  void Push(const T &item) {
    {
      std::lock_guard<std::mutex> g(mutex_);
      q_.emplace_back(item);
    }
    cv_.notify_one();
  }

  void Push(T &&item) {
    {
      std::lock_guard<std::mutex> g(mutex_);
      q_.emplace_back(std::move(item));
    }
    cv_.notify_one();
  }

  template <typename U>
  void Extend(const U &items) {
    {
      std::lock_guard<std::mutex> g(mutex_);
      for (auto &item : items) {
        q_.emplace_back(item);
      }
    }
    cv_.notify_all();
  }

  template <typename U>
  void Extend(U &&items) {
    {
      std::lock_guard<std::mutex> g(mutex_);
      for (auto &item : items) {
        q_.emplace_back(std::move(item));
      }
    }
    cv_.notify_all();
  }

  std::deque<T> PopAll(size_t ms, bool *timeout) {
    auto time =
        std::chrono::system_clock::now() + std::chrono::milliseconds(ms);
    std::unique_lock<std::mutex> lock(mutex_);
    *timeout = !cv_.wait_until(lock, time, [this] { return !q_.empty(); });
    std::deque<T> ret;
    if (!*timeout) {
      std::swap(ret, q_);
    }
    return ret;
  }

  T Pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [=] { return !q_.empty(); });
    T rc(std::move(q_.front()));
    q_.pop_front();
    return rc;
  }

  void Pop(T *t) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [=] { return !q_.empty(); });
    *t = std::move(q_.front());
    q_.pop_front();
  }

  size_t Size() {
    std::lock_guard<std::mutex> lock(mutex_);
    return q_.size();
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::deque<T>().swap(q_);
  }

  void SetBlockSize(size_t x) {
    std::unique_lock<std::mutex> lock(mutex_);
    block_size_ = x;
  }

  size_t Read(size_t n, T* vec) {
    if (n == 0) {
      return 0;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [=] { return !q_.empty(); });
    size_t read_num = 0;
    while (read_num < n) {
      size_t m = std::min(n - read_num, q_.size());
      for (size_t i = 0; i < m; i++) {
        vec[read_num++] = std::move(q_.front());
        q_.pop_front();
      }
    }
    cv_.notify_one();
    return read_num;
  }

  size_t ReadIntoVec(const std::vector<T> * vec) {
    vec->resize(block_size_);
    size_t read_num = Read(vec->size(), &vec[0]);
    vec.resize(read_num);
    return read_num;
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<T> q_;
  size_t block_size_ = 80000;
};

}  // namespace framework
}  // namespace paddle
