//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <time.h>
#include <fstream>

#include "paddle/fluid/framework/dataset_factory.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace train {

void ReadBinaryFile(const std::string& filename, std::string* contents) {
  std::ifstream fin(filename, std::ios::in | std::ios::binary);
  PADDLE_ENFORCE(static_cast<bool>(fin), "Cannot open file %s", filename);
  fin.seekg(0, std::ios::end);
  contents->clear();
  contents->resize(fin.tellg());
  fin.seekg(0, std::ios::beg);
  fin.read(&(contents->at(0)), contents->size());
  fin.close();
}

std::unique_ptr<paddle::framework::ProgramDesc> LoadProgramDesc(
    const std::string& model_filename) {
  VLOG(3) << "loading model from " << model_filename;
  std::string program_desc_str;
  ReadBinaryFile(model_filename, &program_desc_str);

  std::unique_ptr<paddle::framework::ProgramDesc> main_program(
      new paddle::framework::ProgramDesc(program_desc_str));
  return main_program;
}

}  // namespace train
}  // namespace paddle

int main(int argc, char* argv[]) {
  // filelist, data_feed.prototxt startup_prog, main_prog, model
  std::string filelist = std::string(argv[1]);
  std::vector<std::string> file_vec;
  std::ifstream fin(filelist);
  if (fin) {
    std::string filename;
    while (fin >> filename) {
      file_vec.push_back(filename);
    }
  }

  std::string data_feed_desc_str;
  paddle::train::ReadBinaryFile(std::string(argv[2]), &data_feed_desc_str);

  std::unique_ptr<paddle::framework::Dataset> dataset_ptr;
  // dataset_ptr =
  // paddle::framework::DatasetFactory::CreateDataset("MultiSlotDataset");
  dataset_ptr->SetFileList(file_vec);
  dataset_ptr->SetThreadNum(1);
  dataset_ptr->SetDataFeedDesc(data_feed_desc_str);
  const std::vector<paddle::framework::DataFeed*> readers =
      dataset_ptr->GetReaders();

  // load program here
  const auto cpu_place = paddle::platform::CPUPlace();
  paddle::framework::Executor executor(cpu_place);
  paddle::framework::Scope scope;
  auto startup_program = paddle::train::LoadProgramDesc(std::string(argv[3]));
  auto main_program = paddle::train::LoadProgramDesc(std::string(argv[4]));

  // run startup program
  auto& block = main_program->Block(0);
  executor.Run(*startup_program, &scope, 0);

  paddle::framework::Scope* child_scope = &scope.NewScope();
  for (auto& var : block.AllVars()) {
    if (var->Persistable()) {
      auto* ptr = child_scope->Var(var->Name());
      InitializeVariable(ptr, var->GetType());
    } else {
      auto* ptr = child_scope->Var(var->Name());
      InitializeVariable(ptr, var->GetType());
    }
  }

  std::vector<paddle::framework::OperatorBase*> ops;
  for (auto& op_desc : block.AllOps()) {
    std::unique_ptr<paddle::framework::OperatorBase> local_op =
        paddle::framework::OpRegistry::CreateOp(*op_desc);
    paddle::framework::OperatorBase* local_op_ptr = local_op.release();
    ops.push_back(local_op_ptr);
    continue;
  }

  int epoch_num = 10;

  for (int epoch = 0; epoch < epoch_num; ++epoch) {
    readers[0]->Start();
    int cur_batch = 0;
    while ((cur_batch = readers[0]->Next()) > 0) {
      for (auto& op : ops) {
        op->Run(*child_scope, cpu_place);
      }
      child_scope->DropKids();
    }
  }
}
