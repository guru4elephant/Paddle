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

#include "paddle/fluid/framework/fleet/fleet_wrapper.h"

namespace paddle {
namespace framework {

void FleetWrapper::InitServer(const std::string& dist_desc, int index) {
  _pslib_ptr->init_server(dist_desc, index);
}

void FleetWrapper::InitWorker(const std::string& dist_desc,
                              const std::vector<uint64_t>& host_sign_list,
                              int node_num, int index) {
  _pslib_ptr->init_worker(
      dist_desc, const_cast<uint64_t*>(host_sign_list.data()), node_num, index);
}

void FleetWrapper::StopServer() { _pslib_ptr->stop_server(); }

uint64_t FleetWrapper::RunServer() { return _pslib_ptr->run_server(); }

void FleetWrapper::GatherServers(const std::vector<uint64_t>& host_sign_list,
                                 int node_num) {
  _pslib_ptr->gather_servers(const_cast<uint64_t*>(host_sign_list.data()),
                             node_num);
}

}  // end namespace framework
}  // end namespace paddle
