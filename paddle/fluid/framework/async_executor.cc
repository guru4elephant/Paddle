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

#include "paddle/fluid/framework/async_executor.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"

#include "gflags/gflags.h"
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/executor_thread_worker.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/pybind/pybind.h"
#ifdef PADDLE_WITH_PSLIB
#include <pslib.h>
#endif

namespace paddle {
namespace framework {
AsyncExecutor::AsyncExecutor(Scope* scope, const platform::Place& place)
    : root_scope_(scope), place_(place) {}

void AsyncExecutor::InitServer(const std::string& dist_desc, int index) {
  _fleet_ptr = FleetWrapper::GetInstance();
  _fleet_ptr->InitServer(dist_desc, index);
}

void AsyncExecutor::InitWorker(const std::string& dist_desc,
                               const std::vector<uint64_t>& host_sign_list,
                               int node_num, int index) {
  _fleet_ptr = FleetWrapper::GetInstance();
  _fleet_ptr->InitWorker(dist_desc, host_sign_list, node_num, index);
  InitParamConfig();
}

uint64_t AsyncExecutor::StartServer() { return _fleet_ptr->RunServer(); }

void AsyncExecutor::StopServer() { _fleet_ptr->StopServer(); }

void AsyncExecutor::GatherServers(const std::vector<uint64_t>& host_sign_list,
                                  int node_num) {
  _fleet_ptr->GatherServers(host_sign_list, node_num);
}

void AsyncExecutor::InitParamConfig() {
  /*
  for (int i = 0; i < _pslib_ptr->get_param()
                          ->server_param()
                          .downpour_server_param()
                          .downpour_table_param_size();
       ++i) {
    if (_pslib_ptr->get_param()
            ->server_param()
            .downpour_server_param()
            .downpour_table_param(i)
            .table_class()
            .find("SparseTable") != -1) {
      _param_config.fea_dim = _pslib_ptr->get_param()
                                  ->server_param()
                                  .downpour_server_param()
                                  .downpour_table_param(i)
                                  .accessor()
                                  .fea_dim();
      break;
    }
  }
  _param_config.slot_dim = _param_config.fea_dim - 2;
  _param_config.tmp_push_dense_wait_times = static_cast<int32_t>(
      _pslib_ptr->get_param()->trainer_param().push_dense_per_batch());
  _param_config.tmp_push_sparse_wait_times = static_cast<int32_t>(
      _pslib_ptr->get_param()->trainer_param().push_sparse_per_batch());

  for (auto t = 0u; t < _pslib_ptr->get_param()->trainer_param().skip_op_size();
       ++t) {
    _param_config.skip_op.push_back(
        _pslib_ptr->get_param()->trainer_param().skip_op(t));
  }

  for (auto t = 0u;
       t < _pslib_ptr->get_param()->trainer_param().sparse_table_size(); ++t) {
    auto& table = _pslib_ptr->get_param()->trainer_param().sparse_table(t);
    std::vector<std::string> tmp_sparse_variable_name;
    for (int i = 0u; i < table.slot_value_size(); ++i) {
      tmp_sparse_variable_name.push_back(table.slot_value(i));
      _param_config.slot_alias_to_table[table.slot_key(i)] = table.table_id();
    }
    std::vector<std::string> tmp_sparse_gradient_variable_name;
    for (auto i = 0u; i < table.slot_gradient_size(); ++i) {
      tmp_sparse_gradient_variable_name.push_back(table.slot_gradient(i));
    }
    _param_config.slot_input_vec[table.table_id()] =
        std::move(tmp_sparse_variable_name);
    _param_config.gradient_var[table.table_id()] =
        std::move(tmp_sparse_gradient_variable_name);
    _param_config.sparse_table_id.push_back(table.table_id());
  }

  for (auto t = 0u;
       t < _pslib_ptr->get_param()->trainer_param().dense_table_size(); ++t) {
    auto& table = _pslib_ptr->get_param()->trainer_param().dense_table(t);
    std::vector<std::string> tmp_dense_variable_name;
    for (int i = 0u; i < table.dense_variable_name_size(); ++i) {
      tmp_dense_variable_name.push_back(table.dense_variable_name(i));
    }
    std::vector<std::string> tmp_dense_gradient_variable_name;
    for (auto i = 0u; i < table.dense_gradient_variable_name_size(); ++i) {
      tmp_dense_gradient_variable_name.push_back(
          table.dense_gradient_variable_name(i));
    }
    _param_config.dense_variable_name[table.table_id()] =
        std::move(tmp_dense_variable_name);
    _param_config.dense_gradient_variable_name[table.table_id()] =
        std::move(tmp_dense_gradient_variable_name);
    _param_config.dense_table_id.push_back(table.table_id());
    _param_config.dense_table_size.push_back(table.fea_dim());
  }
  */
}

void AsyncExecutor::InitModel() {
  /*
  for (auto table_id : _param_config.dense_table_id) {
    std::vector<paddle::ps::Region> regions;
    for (auto& t : _param_config.dense_variable_name[table_id]) {
      Variable* var = root_scope_->FindVar(t);
      CHECK(var != nullptr) << "var[" << t << "] not found";
      LoDTensor* tensor = var->GetMutable<LoDTensor>();

      float* g = tensor->data<float>();
      CHECK(g != nullptr) << "var[" << t << "] value not initialized";

      float init_range = 0.2;
      int rown = tensor->dims()[0];
      init_range /= sqrt(rown);

      std::normal_distribution<float> ndistr(0.0, 1.0);
      for (auto i = 0u; i < tensor->numel(); ++i) {
        g[i] = ndistr(local_random_engine()) * init_range;
      }

      paddle::ps::Region reg(g, tensor->numel());
      regions.emplace_back(std::move(reg));
    }

    auto push_status = _pslib_ptr->_worker_ptr->push_dense_param(
        regions.data(), regions.size(), table_id);
    push_status.wait();
    auto status = push_status.get();
    if (status != 0) {
      LOG(FATAL) << "push dense param failed, status[" << status << "]";
      exit(-1);
    }
  }
  */
}

void AsyncExecutor::SaveModel(const std::string& path) {
  /*
  auto ret = _pslib_ptr->_worker_ptr->flush();
  ret.wait();
  ret = _pslib_ptr->_worker_ptr->save(path, 0);
  ret.wait();
  int32_t feasign_cnt = ret.get();
  if (feasign_cnt == -1) {  // (colourful-tree) TODO should be feasign_cnt < 0
    LOG(FATAL) << "save model failed";
    exit(-1);
  }
  */
}

void AsyncExecutor::PrepareDenseThread(const std::string& mode) {
  /*
  if (mode == "mpi") {
    DensePullThreadParam param;
    param.ps_client = _pslib_ptr->_worker_ptr;
    param.threshold = 1;
    param.training_thread_num = actual_thread_num;
    param.root_scope = root_scope_;
    param.dense_params = &_param_config.dense_variable_name;

    _pull_dense_thread =
        std::shared_ptr<DensePullThread>(new DensePullThread(param));
    _pull_dense_thread->start();
  }
  */
}

void AsyncExecutor::RunFromFile(const ProgramDesc& main_program,
                                const std::string& trainer_desc_str,
                                const bool debug) {
  TrainerDesc trainer_desc;
  google::protobuf::TextFormat::ParseFromString(trainer_desc_str,
                                                &trainer_desc);
  std::shared_ptr<TrainerBase> trainer;
  trainer = TrainerFactory::CreateTrainer(trainer_desc.class_name());
  // initialize trainer
  trainer->Initialize(trainer_desc);
  // trainer->SetRootScope(root_scope_);
  trainer->SetDebug(debug);
  // prepare training environment and helper environment
  trainer->InitTrainerEnv(main_program, place_);
  trainer->InitOtherEnv(main_program);
  // training and finalize training
  trainer->Run();
  trainer->Finalize();
  root_scope_->DropKids();
  return;
}

}  // end namespace framework
}  // end namespace paddle
