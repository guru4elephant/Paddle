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

#include <memory>
#ifdef PADDLE_WITH_PSLIB
#include <archive.h>
#include <pslib.h>
#endif
#include <atomic>
#include <ctime>
#include <map>
#include <random>
#include <string>
#include <vector>
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/macros.h"  // for DISABLE_COPY_AND_ASSIGN

namespace paddle {
namespace framework {

// A wrapper class for pslib.h, this class follows Singleton pattern
// i.e. only initialized once in the current process
// Example:
//    std::shared_ptr<FleetWrapper> fleet_ptr =
//         FleetWrapper::GetInstance();
//    string dist_desc;
//    fleet_ptr->InitServer(dist_desc, 0);
// interface design principles:
// Pull
//   Sync: PullSparseVarsSync
//   Async: PullSparseVarsAsync(not implemented currently)
// Push
//   Sync: PushSparseVarsSync
//   Async: PushSparseVarsAsync(not implemented currently)
//   Async: PushSparseVarsWithLabelAsync(with special usage)
// Push dense variables to server in Async mode
// Param<in>: scope, table_id, var_names
// Param<out>: push_sparse_status

class FleetWrapper {
 public:
  virtual ~FleetWrapper() {}
  FleetWrapper() {}
  // Pull sparse variables from server in Sync mode
  // Param<in>: scope, table_id, var_names, fea_keys
  // Param<out>: fea_values
  void PullSparseVarsSync(const Scope& scope, const uint64_t table_id,
                          const std::vector<std::string>& var_names,
                          std::vector<uint64_t>* fea_keys,
                          std::vector<std::vector<float>>* fea_values,
                          int fea_dim);

  void PullDenseVarsSync(const Scope& scope, const uint64_t table_id,
                         const std::vector<std::string>& var_names);

  void PullDenseVarsAsync(
      const Scope& scope, const uint64_t table_id,
      const std::vector<std::string>& var_names,
      std::vector<::std::future<int32_t>>* pull_dense_status);

  void PushDenseParamSync(const Scope& scope, const uint64_t table_id,
                          const std::vector<std::string>& var_names);

  // Push dense variables to server in async mode
  // Param<in>: scope, table_id, var_names,
  // Param<out>: push_sparse_status
  void PushDenseVarsAsync(
      const Scope& scope, const uint64_t table_id,
      const std::vector<std::string>& var_names,
      std::vector<::std::future<int32_t>>* push_sparse_status);

  void PushDenseVarsSync(Scope* scope, const uint64_t table_id,
                         const std::vector<std::string>& var_names);

  // Push sparse variables with labels to server in Async mode
  // This is specially designed for click/show stats in server
  // Param<in>: scope, table_id, var_grad_names,
  //            fea_keys, fea_labels, sparse_grad_names
  // Param<out>: push_values, push_sparse_status
  void PushSparseVarsWithLabelAsync(
      const Scope& scope, const uint64_t table_id,
      const std::vector<uint64_t>& fea_keys,
      const std::vector<float>& fea_labels,
      const std::vector<std::string>& sparse_key_names,
      const std::vector<std::string>& sparse_grad_names, const int emb_dim,
      std::vector<std::vector<float>>* push_values,
      std::vector<::std::future<int32_t>>* push_sparse_status,
      int cur_batch);

  // Push sparse variables to server in Async mode
  // Param<In>: scope, table_id, fea_keys, sparse_grad_names
  // Param<Out>: push_values, push_sparse_status
  /*
  void PushSparseVarsAsync(
          const Scope& scope,
          const uint64_t table_id,
          const std::vector<uint64_t>& fea_keys,
          const std::vector<std::string>& sparse_grad_names,
          std::vector<std::vector<float>>* push_values,
          std::vector<::std::future<int32_t>>* push_sparse_status);
  */

  void InitServer(const std::string& dist_desc, int index);
  void InitWorker(const std::string& dist_desc,
                  const std::vector<uint64_t>& host_sign_list, int node_num,
                  int index);
  void StopServer();
  uint64_t RunServer();
  void GatherServers(const std::vector<uint64_t>& host_sign_list, int node_num);
  // gather client ip
  void GatherClients(const std::vector<uint64_t>& host_sign_list);
  // get client info
  std::vector<uint64_t> GetClientsInfo();
  // create client to client connection
  void CreateClient2ClientConnection();


  void LoadModel(const std::string& path,
                               const std::string& mode) {
    auto ret = pslib_ptr_->_worker_ptr->load(path, mode);
    ret.wait();
    if (ret.get() != 0) {
      LOG(ERROR) << "load model from path:" << path << " failed";
      exit(-1);
    }
  }

  void ServerFlush() {
    auto ret = pslib_ptr_->_worker_ptr->flush();
    ret.wait();
  }

  // param = 0, save all feature
  // param = 1, save delta feature
  void SaveModel(const std::string& path, const std::string& mode) {
    //auto ret = pslib_ptr_->_worker_ptr->flush();
    //ret.wait();
    auto ret = pslib_ptr_->_worker_ptr->save(path, mode);
    ret.wait();
    int32_t feasign_cnt = ret.get();
    if (feasign_cnt == -1) {
        LOG(FATAL) << "save model failed";
        exit(-1);
    }
  }

  void ShrinkSparseTable(int table_id) {
    auto ret = pslib_ptr_->_worker_ptr->shrink(table_id);
    ret.wait();
  }

  void ShrinkDenseTable(int table_id, Scope* scope, std::vector<std::string> var_list, float decay) {
    std::vector<paddle::ps::Region> regions;
    for (std::string& name : var_list) {
      if (name.find("batch_sum") != std::string::npos) {
Variable* var = scope->FindVar(name);
                    CHECK(var != nullptr) << "var[" << name << "] not found";
                    LOG(ERROR) << "prepare shrink dense batch_sum";
                    LoDTensor* tensor = var->GetMutable<LoDTensor>();
                    float* g = tensor->data<float>();
                    //CHECK(g != nullptr) << "var[" << t << "] value not initialized";
                    Eigen::Map<Eigen::MatrixXf> mat(g, 1, tensor->numel());
                    
                    mat *= decay;
                    paddle::ps::Region reg(g, tensor->numel());
                    regions.emplace_back(std::move(reg));
      } else {
           Variable* var = scope->FindVar(name);
                    CHECK(var != nullptr) << "var[" << name << "] not found";
                    LoDTensor* tensor = var->GetMutable<LoDTensor>();
                    float* g = tensor->data<float>();
                    //CHECK(g != nullptr) << "var[" << t << "] value not initialized";
                    paddle::ps::Region reg(g, tensor->numel());
                    regions.emplace_back(std::move(reg));  
     // }
    }
auto push_status = pslib_ptr_->_worker_ptr->push_dense_param(regions.data(), regions.size(), table_id);
                push_status.wait();
                auto status = push_status.get();
                if (status != 0) {
                    LOG(FATAL) << "push shrink dense param failed, status[" << status << "]";
                    exit(-1);
                }
  }
  }

  // register client to client communication
  typedef std::function<int32_t(int, int, const std::string&)> MsgHandlerFunc;
  int RegisterClientToClientMsgHandler(int msg_type, MsgHandlerFunc handler);
  // send client to client message
  std::future<int32_t> SendClientToClientMsg(int msg_type, int to_client_id,
                                             const std::string& msg);

  template <typename T>
  void Serialize(const std::vector<T>& t,
                 const size_t begin,
                 const size_t end,
                 std::string* str);

  template <typename T>
  void Serialize(const std::vector<T>& t, std::string* str);

  template <typename T>
  void Deserialize(std::vector<T>* t, const std::string& str);
  static std::shared_ptr<FleetWrapper> GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::framework::FleetWrapper());
    }
    return s_instance_;
  }

//  double current_realtime() {
//    struct timespec tp;
//    clock_gettime(CLOCK_REALTIME, &tp);
//    return tp.tv_sec + tp.tv_nsec * 1e-9;
//  }
/*
  std::default_random_engine& LocalRandomEngine() {
    //struct timespec tp;
    //clock_gettime(CLOCK_REALTIME, &tp);
    //double current_realtime = tp.tv_sec + tp.tv_nsec * 1e-9;
    double current_realtime() {
            struct timespec tp;
                clock_gettime(CLOCK_REALTIME, &tp);
                    return tp.tv_sec + tp.tv_nsec * 1e-9;
    }
    struct engine_wrapper_t {
        //struct timespec tp;
        //clock_gettime(CLOCK_REALTIME, &tp);
        //double current_realtime = tp.tv_sec + tp.tv_nsec * 1e-9;
        std::default_random_engine engine;
        engine_wrapper_t() {
            static std::atomic<unsigned long> x(0);
            std::seed_seq sseq = {x++, x++, x++, (unsigned long)(current_realtime() * 1000)};
            engine.seed(sseq);
        }
    };
    thread_local engine_wrapper_t r;
    return r.engine;
}
*/
std::default_random_engine& LocalRandomEngine() {
  struct engine_wrapper_t {
    std::default_random_engine engine;
    engine_wrapper_t() {
      struct timespec tp;
      clock_gettime(CLOCK_REALTIME, &tp);
      double cur_time = tp.tv_sec + tp.tv_nsec * 1e-9;
      static std::atomic<uint64_t> x(0);
      std::seed_seq sseq = {x++, x++, x++, (uint64_t)(cur_time * 1000)};
      engine.seed(sseq);
    }
  };
  thread_local engine_wrapper_t r;
  return r.engine;
}


#ifdef PADDLE_WITH_PSLIB
  static std::shared_ptr<paddle::distributed::PSlib> pslib_ptr_;
#endif

 private:
  static std::shared_ptr<FleetWrapper> s_instance_;
#ifdef PADDLE_WITH_PSLIB
  std::map<uint64_t, std::vector<paddle::ps::Region>> _regions;
#endif

 protected:
  static bool is_initialized_;
  DISABLE_COPY_AND_ASSIGN(FleetWrapper);
};

/*double current_realtime() {
    struct timespec tp;
    clock_gettime(CLOCK_REALTIME, &tp);
    return tp.tv_sec + tp.tv_nsec * 1e-9;
}

std::default_random_engine& local_random_engine() {
    struct engine_wrapper_t {
        std::default_random_engine engine;
        engine_wrapper_t() {
            static std::atomic<unsigned long> x(0);
            std::seed_seq sseq = {x++, x++, x++, (unsigned long)(current_realtime() * 1000)};
            engine.seed(sseq);
        }
    };
    thread_local engine_wrapper_t r;
    return r.engine;
}*/

}  // end namespace framework
}  // end namespace paddle
