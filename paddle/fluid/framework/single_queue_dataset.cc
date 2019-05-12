/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License. */

#include <random>
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/io/fs.h"
#include "paddle/fluid/platform/timer.h"

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

namespace paddle {
namespace framework {


void SingleQueueDatasetImpl::LocalShuffle() {
  return;
}

void SingleQueueDatasetImpl::CreateReaders() {
  for (int i = 0; i < thread_num_; ++i) {
    readers_.push_back(DataFeedFactory::CreateDataFeed(data_feed_desc_.name()));
    readers_.back()->Init(data_feed_desc_);
    readers_.back()->SetInputQueue(read_queue_);
    readers_.back()->SetOutputQueue(write_queue_);
  }
}

void SingleQueueDatasetImpl::GlobalShuffle() {
  VLOG(3) << "GlobalShuffle() begins";
  auto fleet_ptr = FleetWrapper::GetInstance();
  // suppose our input queue has N instances
  // repeat:
  //   get fleet_send_batch_size_ from input queue
  //   repeat:
  //     send_len_each_worker instances to a random worker
  //   until all instances in current send_batch are consumed
  // until all instances in input queue has been consumed
  std::vector<int> send_index(trainer_num_);
  for (int i = 0; i < trainer_num_; ++i) {
    send_index[i] = i;
  }
  std::vector<std::vector<MultiSlotType>> send_block;
  std::vector<std::vector<MultiSlotType>>::iterator begin;
  std::vector<std::vector<MultiSlotType>>::iterator end;
  std::vector<std::future<int32_t>> total_status;
  read_queue_->SetBlockSize(fleet_send_batch_size_);
  while (read_queue_->Size() > 0) {
    std::shuffle(send_index.begin(),
                 send_index.end(),
                 fleet_ptr->LocalRandomEngine());
    // read a block here
    uint64_t actual_read_size = read_queue_->ReadIntoVec(&send_block);
    uint64_t send_len_each_worker =
        actual_read_size / trainer_num_ + 1;
    std::shuffle(
        send_block.begin(),
        send_block.end(),
        fleet_ptr->LocalRandomEngine());
    
    for (int index = 0; index < send_index.size(); ++index) {
      int target_node = send_index[index];
      std::string send_str;
      size_t begin = index * send_len_each_worker;
      size_t end = begin + std::min(
          send_len_each_worker,
          send_block.size() - begin);
      SerializeIns(send_block, begin, end, &send_str);
      auto ret = fleet_ptr->SendClientToClientMsg(0, target_node, send_str);
      total_status.push_back(std::move(ret));
    }

    for (auto& t : total_status) {
      t.wait();
    }
    send_block.clear();
  }
}

void SingleQueueDatasetImpl::LoadIntoMemory() {
  // set local queue here
  std::vector<std::thread> load_threads;
  for (int64_t i = 0; i < thread_num_; ++i) {
    /*
    load_threads.push_back(std::thread(
        &paddle::framework::MultiSlotDataFeed::ReadThread,
        readers_[i].get()));
    */
    continue;
  }
  
  for (std::thread& t : load_threads) {
    t.join();
  }
}

void SingleQueueDatasetImpl::ReleaseMemory() {
  return;
}


}  // end namespace framework
}  // end namespace paddle
