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

std::shared_ptr<FleetWrapper> FleetWrapper::_s_instance = NULL;

void FleetWrapper::InitServer(const std::string& dist_desc, int index) {
#ifdef PADDLE_WITH_PSLIB
  if (_is_initialized) {
    LOG(WARNING) << "Server has been initialized, "
                    "can not do this more than twice";
  } else {
    _pslib_ptr->init_server(dist_desc, index);
    _is_initialized = true;
  }
#endif
  return;
}

void FleetWrapper::InitWorker(const std::string& dist_desc,
                              const std::vector<uint64_t>& host_sign_list,
                              int node_num, int index) {
#ifdef PADDLE_WITH_PSLIB
  if (_is_initialized) {
    LOG(WARNING) << "Worker has been initialized, "
                    "can not do this more than twice";
  } else {
    _pslib_ptr->init_worker(dist_desc,
                            const_cast<uint64_t*>(host_sign_list.data()),
                            node_num, index);
    _is_initialized = true;
  }
#endif
  return;
}

void FleetWrapper::StopServer() {
#ifdef PADDLE_WITH_PSLIB
  if (!_is_initialized) {
    LOG(WARNING) << "Server has not been initialized, "
                    "can not do this";
  } else {
    _pslib_ptr->stop_server();
  }
#endif
}

uint64_t FleetWrapper::RunServer() {
#ifdef PADDLE_WITH_PSLIB
  if (!_is_initialized) {
    LOG(WARNING) << "Server has not been initialized, "
                    "can not run server";
  } else {
    return _pslib_ptr->run_server();
  }
#else
  return 0;
#endif
}

void FleetWrapper::GatherServers(const std::vector<uint64_t>& host_sign_list,
                                 int node_num) {
#ifdef PADDLE_WITH_PSLIB
  if (_is_initialized) {
    LOG(WARNING) << "Server has been initialized, "
                    "can not do this";
  } else {
    _pslib_ptr->gather_servers(const_cast<uint64_t*>(host_sign_list.data()),
                               node_num);
  }
#endif
}

}  // end namespace framework
}  // end namespace paddle
