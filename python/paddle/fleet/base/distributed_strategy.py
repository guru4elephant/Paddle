#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle.fleet.proto import distributed_strategy_pb2
from paddle.fluid.framework import Variable
import google.protobuf.text_format


class DistributedJobInfo(object):
    """
    DistributedJobInfo will serialize all distributed training information
    Just for inner use: 1) debug 2) replicate experiments
    """

    def __init__(self):
        self.job_info = distributed_strategy_pb2.DistributedJobInfo()

    def _set_worker_num(self, worker_num):
        self.job_info.worker_num = worker_num

    def _set_server_num(self, server_num):
        self.job_info.server_num = server_num

    def _set_worker_ips(self, worker_ips):
        self.job_info.worker_ips.extend(worker_ips)

    def _set_server_endpoints(self, server_endpoints):
        self.job_info.server_endpoints.extend(server_endpoints)

    def _set_origin_startup(self, origin_startup_prog):
        self.job_info.origin_startup = str(origin_startup_prog)

    def _set_origin_main(self, origin_main_prog):
        self.job_info.origin_main = str(origin_main_prog)

    def _distributed_main(self, distributed_main_prog):
        self.job_info.distributed_main = str(distributed_main_prog)

    def _optimizer_name(self, optimizer_name):
        self.job_info.optimizer_name = optimizer_name

    def _set_distributed_strategy(self, dist_strategy):
        self.job_info.strategy = dist_strategy


class DistributedStrategy(object):
    def __init__(self):
        self.strategy = distributed_strategy_pb2.DistributedStrategy()

    def save_to_prototxt(self, output):
        with open(output, "w") as fout:
            fout.write(str(self.strategy))

    def load_from_prototxt(self, pb_file):
        f = open(pb_file, 'r')
        self.strategy = google.protobuf.text_format.Merge(
            str(f.read()), self.strategy)

    @property
    def execution_strategy(self):
        # return a strategy dict
        return None

    @execution_strategy.setter
    def execution_strategy(self, strategy):
        self.execution_strategy.num_iteration_per_drop_scope = \
                    strategy.num_iteration_per_drop_scope
        self.execution_strategy.num_iteration_per_run = \
                    strategy.num_iteration_per_run
        self.execution_strategy.num_threads = \
                    strategy.num_threads

    @property
    def build_strategy(self):
        # return a strategy dict
        build_strategy = paddle.fluid.BuildStrategy()
        build_strategy.enable_sequential_execution = \
                    self.strategy.enable_sequential_execution
        build_strategy.fuse_broadcast_ops = \
                    self.strategy.fuse_broadcast_ops
        build_strategy.fuse_elewise_add_act_ops = \
                    self.strategy.fuse_elewise_add_act_ops
        build_strategy.fuse_relu_depthwise_conv = \
                    self.strategy.fuse_relu_depthwise_conv
        build_strategy.sync_batch_norm = \
                    self.strategy.sync_batch_norm
        build_strategy.nccl_comm_num = \
                    self.strategy.nccl_comm_num
        build_strategy.use_hierarchical_allreduce = \
                    self.strategy.use_hierarchical_allreduce
        build_strategy.hierarchical_allreduce_inter_nranks = \
                    self.strategy.hierarchical_allreduce_inter_nranks
        build_strategy.fuse_all_reduce_ops = \
                    self.strategy.fuse_all_reduce_ops
        return build_strategy

    @build_strategy.setter
    def build_strategy(self, strategy):
        self.strategy.enable_sequential_execution = \
                    strategy.enable_sequential_execution
        self.strategy.fuse_broadcast_ops = \
                    strategy.fuse_broadcast_ops
        self.strategy.fuse_elewise_add_act_ops = \
                    strategy.fuse_elewise_add_act_ops
        self.strategy.fuse_relu_depthwise_conv = \
                    strategy.fuse_relu_depthwise_conv
        self.strategy.sync_batch_norm = \
                    strategy.sync_batch_norm
        self.strategy.nccl_comm_num = \
                    strategy.nccl_comm_num
        self.strategy.use_hierarchical_allreduce = \
                    strategy.use_hierarchical_allreduce
        self.strategy.hierarchical_allreduce_inter_nranks = \
                    strategy.hierarchical_allreduce_inter_nranks
        self.strategy.fuse_all_reduce_ops = \
                    strategy.fuse_all_reduce_ops

    @property
    def amp(self):
        return self.strategy.amp

    @amp.setter
    def amp(self, flag):
        if isinstance(flag, bool):
            self.strategy.amp = flag
        else:
            print("WARNING: amp should have value of bool type")

    @property
    def amp_configs(self):
        return self.amp_configs

    @amp_configs.setter
    def amp_configs(self, configs):
        self.amp_configs = configs
        self.strategy.amp_loss_scaling = configs["amp_loss_scaling"]

    @property
    def recompute(self):
        return self.strategy.recompute

    @recompute.setter
    def recompute(self, flag):
        if isinstance(flag, bool):
            self.strategy.recompute = flag
        else:
            print("WARNING: recompute should have value of bool type")

    @property
    def recompute_configs(self):
        configs = {}
        configs["recompute_checkpoints"] = self.strategy.recompute_checkpoints
        return configs

    def recompute_configs(self, configs):
        self.strategy.recompute_configs = configs
        self.strategy.recompute_checkpoints = configs["recompute_checkpoints"]

    @property
    def pipeline(self):
        return self.strategy.pipeline

    @pipeline.setter
    def pipeline(self, flag):
        if isinstance(flag, bool):
            self.strategy.pipeline = flag
        else:
            print("WARNING: pipeline should have value of bool type")

    @property
    def pipeline_configs(self):
        configs = {}
        configs["pipeline_macro_batch"] = \
            self.strategy.pipeline_macro_batch

    @pipeline_configs.setter
    def pipeline_configs(self, configs):
        self.pipeline_configs = configs
        self.strategy.pipeline_micro_batch = configs["pipeline_macro_batch"]

    @property
    def localsgd(self):
        return self.strategy.localsgd

    @localsgd.setter
    def localsgd(self, flag):
        if isinstance(flag, bool):
            self.strategy.localsgd = flag
        else:
            print("WARNING: localsgd should have value of bool type")

    @property
    def localsgd_configs(self):
        configs = {}
        configs["localsgd_k_step"] = \
            self.strategy.localsgd_k_step
        return configs

    @localsgd.setter
    def localsgd_configs(self, configs):
        self.strategy.localsgd_k_step = configs["k_step"]

    @property
    def dgc(self):
        return self.strategy.dgc

    @dgc.setter
    def dgc(self, flag):
        if isinstance(flag, bool):
            self.strategy.dgc = flag
        else:
            print("WARNING: dgc should have value of bool type")

    @property
    def hierachical_allreduce(self):
        return self.strategy.hierachical_allreduce

    @hierachical_allreduce.setter
    def hierachical_allreduce(self, flag):
        if isinstance(flag, bool):
            self.strategy.hierachical_allreduce = flag
        else:
            print(
                "WARNING: hierachical_allreduce should have value of bool type")

    @property
    def hierachical_allreduce_inter_ranks(self):
        return self.strategy.hierachical_allreduce_inter_ranks

    @hierachical_allreduce_inter_ranks.setter
    def hierachical_allreduce_inter_ranks(self, flag):
        if isinstance(flag, bool):
            self.strategy.hierachical_allreduce_inter_ranks = flag
        else:
            print(
                "WARNING: hierachical_allreduce_inter_ranks should have value of bool type"
            )

    @property
    def nccl_comm_num(self):
        return self.strategy.nccl_comm_num

    @nccl_comm_num.setter
    def nccl_comm_num(self, value):
        if isinstance(value, int):
            self.strategy.nccl_comm_num = value
        else:
            print("WARNING: nccl_comm_num should have value of int type")

    @property
    def gradient_merge(self):
        return self.strategy.gradient_merge

    @gradient_merge.setter
    def gradient_merge(self, flag):
        if isinstance(flag, bool):
            self.strategy.gradient_merge = flag
        else:
            print("WARNING: gradient_merge should have value of bool type")

    @property
    def gradient_merge_k_step(self):
        return self.strategy.gradient_merge_k_step

    @gradient_merge_k_step.setter
    def gradient_merge_k_step(self, value):
        if isinstance(value, int):
            self.strategy.gradient_merge_k_step = value
        else:
            print(
                "WARNING: gradient_merge_k_step should have value of int type")

    @property
    def sequential_execution(self):
        return self.strategy.sequential_execution

    @sequential_execution.setter
    def sequential_execution(self, flag):
        if isinstance(flag, bool):
            self.strategy.sequential_execution = flag
        else:
            print(
                "WARNING: sequential_execution should have value of bool type")

    @property
    def sync_nccl_allreduce(self):
        return self.strategy.sync_nccl_allreduce

    @sync_nccl_allreduce.setter
    def sync_nccl_allreduce(self, flag):
        if isinstance(flag, bool):
            self.strategy.sync_nccl_allreduce = flag
        else:
            print("WARNING: sync_nccl_allreduce should have avlue of bool type")

    @property
    def lars(self):
        return self.strategy.lars

    @lars.setter
    def lars(self, flag):
        if isinstance(flag, bool):
            self.strategy.lars = flag
        else:
            print("WARNING: lars should have value of bool type")

    @property
    def lamb(self):
        return self.strategy.lamb

    @lamb.setter
    def lamb(self, flag):
        if isinstance(flag, bool):
            self.strategy.lamb = flag
        else:
            print("WARNING: lamb should have value of bool type")

    @property
    def fuse_elewise_add_act_ops(self):
        return self.strategy.fuse_elewise_add_act_ops

    @fuse_elewise_add_act_ops.setter
    def fuse_elewise_add_act_ops(self, flag):
        if isinstance(flag, bool):
            self.strategy.fuse_elewise_add_act_ops = flag
        else:
            print(
                "WARNING: fuse_elewise_add_act_ops should have value of bool type"
            )

    @property
    def fuse_bn_act_ops(self):
        return self.strategy.fuse_bn_act_ops

    @fuse_bn_act_ops.setter
    def fuse_bn_act_ops(self, flag):
        if isinstance(flag, bool):
            self.strategy.fuse_bn_act_ops = flag
        else:
            print("WARNING: fuse_bn_act_ops should have value of bool type")

    @property
    def enable_auto_fusion(self):
        return self.strategy.enable_auto_fusion

    @enable_auto_fusion.setter
    def enable_auto_fusion(self, flag):
        if isinstance(flag, bool):
            self.strategy.enable_auto_fusion = flag
        else:
            print("WARNING: enable_auto_fusion should have value of bool type")

    @property
    def fuse_relu_depthwise_conv(self):
        return self.strategy.fuse_relu_depthwise_conv

    @fuse_relu_depthwise_conv.setter
    def fuse_relu_depthwise_conv(self, flag):
        if isinstance(flag, bool):
            self.strategy.fuse_relu_depthwise_conv = flag
        else:
            print(
                "WARNING: fuse_relu_depthwise_conv should have value of bool type"
            )

    @property
    def fuse_broadcast_ops(self):
        return self.strategy.fuse_broadcast_ops

    @fuse_broadcast_ops.setter
    def fuse_broadcast_ops(self, flag):
        if isinstance(flag, bool):
            self.strategy.fuse_broadcast_ops = flag
        else:
            print("WARNING: fuse_broadcast_ops should have value of bool type")

    @property
    def enable_inplace(self):
        return self.strategy.enable_inplace

    @enable_inplace.setter
    def enable_inplace(self, flag):
        if isinstance(flag, bool):
            self.strategy.enable_inplace = flag
        else:
            print("WARNING: enable_inplace should have value of bool type")

    @property
    def fuse_all_reduce_ops(self):
        return self.strategy.fuse_all_reduce_ops

    @fuse_all_reduce_ops.setter
    def fuse_all_reduce_ops(self, flag):
        if isinstance(flag, bool):
            self.strategy.fuse_all_reduce_ops = flag
        else:
            print("WARNING: fuse_all_reduce_ops should have value of bool type")

    @property
    def num_iteration_per_drop_scope(self):
        return self.strategy.num_iteration_per_drop_scope

    @num_iteration_per_drop_scope.setter
    def num_iteration_per_drop_scope(self, flag):
        if isinstance(flag, int):
            self.strategy.num_iteration_per_drop_scope = flag
        else:
            print(
                "WARNING: num_iteration_per_drop_scope should have value of int type"
            )

    @property
    def num_iteration_per_run(self):
        return self.strategy.num_iteration_per_run

    @num_iteration_per_run.setter
    def num_iteration_per_run(self, value):
        if isinstance(value, int):
            self.strategy.num_iteration_per_run = value
        else:
            print(
                "WARNING: num_iteration_per_run should have value of int type")

    @property
    def sync_batch_norm(self):
        return self.strategy.sync_batch_norm

    @sync_batch_norm.setter
    def sync_batch_norm(self, flag):
        if isinstance(flag, bool):
            self.strategy.sync_batch_norm = flag
        else:
            print("WARNING: sync_batch_norm should have value of bool type")

    @property
    def fuse_all_optimizer_ops(self):
        return self.strategy.fuse_all_optimizer_ops

    @fuse_all_optimizer_ops.setter
    def fuse_all_optimizer_ops(self, flag):
        if isinstance(flag, bool):
            self.strategy.fuse_all_optimizer_ops = flag
        else:
            print(
                "WARNING: fuse_all_optimizer_ops should have value of bool type")

    @property
    def sync(self):
        return self.strategy.sync

    @sync.setter
    def sync(self, flag):
        if isinstance(flag, bool):
            self.strategy.sync = flag
        else:
            print("WARNING: sync should have value of bool type")

    @property
    def async_k_step(self):
        return self.strategy.async_k_step

    @async_k_step.setter
    def async_k_step(self, value):
        if isinstance(value, int):
            self.strategy.async_k_step = value
        else:
            print("WARNING: async_k_step should have value of int type")

    @property
    def max_merge_var_num(self):
        return self.strategy.max_merge_var_num

    @max_merge_var_num.setter
    def max_merge_var_num(self, value):
        if isinstance(value, int):
            self.strategy.max_merge_var_num = value
        else:
            print("WARNING: max_merge_var_num should have value of int type")

    @property
    def send_queue_size(self):
        return self.strategy.send_queue_size

    @send_queue_size.setter
    def send_queue_size(self, value):
        if isinstance(value, int):
            self.strategy.send_queue_size = value
        else:
            print("WARNING: send_queue_size should have value of int type")

    @property
    def independent_recv_thread(self):
        return self.strategy.independent_recv_thread

    @independent_recv_thread.setter
    def independent_recv_thread(self, value):
        if isinstance(value, bool):
            self.strategy.independent_recv_thread = value
        else:
            print(
                "WARNING: independent_recv_thread should have value of int type")

    @property
    def min_send_grad_num_before_recv(self):
        return self.strategy.min_send_grad_num_before_recv

    @min_send_grad_num_before_recv.setter
    def min_send_grad_num_before_recv(self, value):
        if isinstance(value, int):
            self.strategy.min_send_grad_num_before_recv = value
        else:
            print(
                "WARNING: min_send_grad_num_before_recv should have value of int type"
            )

    @property
    def thread_pool_size(self):
        return self.strategy.thread_pool_size

    @thread_pool_size.setter
    def thread_pool_size(self, value):
        if isinstance(value, int):
            self.strategy.thread_pool_size = value
        else:
            print("WARNING:thread_pool_size should have value of int type")

    @property
    def send_wait_times(self):
        return self.strategy.send_wait_times

    @send_wait_times.setter
    def send_wait_times(self, value):
        if isinstance(value, int):
            self.strategy.send_wait_times = value
        else:
            print("WARNING: send_wait_times should have value of int type")

    @property
    def runtime_split_send_recv(self):
        return self.strategy.runtime_split_send_recv

    @runtime_split_send_recv.setter
    def runtime_split_send_recv(self, flag):
        if isinstance(flag, bool):
            self.strategy.runtime_split_send_recv = flag
        else:
            print("WARNING: runtime_split_send_recv should be bool type")

    @property
    def use_thread_barrier(self):
        return self.strategy.use_thread_barrier

    @use_thread_barrier.setter
    def use_thread_barrier(self, flag):
        if isinstance(flag, bool):
            self.strategy.use_thread_barrier = flag
        else:
            print("WARNING: use_thread_barrier should be bool type")

    @property
    def enable_backward_optimizer_op_deps(self):
        return self.strategy.enable_backward_optimizer_op_deps

    @enable_backward_optimizer_op_deps.setter
    def enable_backward_optimizer_op_deps(self, flag):
        if isinstance(flag, bool):
            self.strategy.enable_backward_optimizer_op_deps = flag
        else:
            print(
                "WARNING: enable_backward_optimizer_op_deps should be bool type")

    @property
    def elastic(self):
        return self.strategy.elastic

    @elastic.setter
    def elastic(self, flag):
        if isinstance(flag, bool):
            self.strategy.elastic = flag
        else:
            print("WARNING: elastic should have value of bool type")

    @property
    def num_threads(self):
        return self.strategy.num_threads

    @num_threads.setter
    def num_threads(self, value):
        if isinstance(value, int):
            self.strategy.num_threads = value
        else:
            print("WARNING: num_threads should have value of int type")

    @property
    def auto(self):
        return self.strategy.auto

    @auto.setter
    def auto(self, flag):
        if isinstance(flag, bool):
            self.strategy.auto = flag
        else:
            print("WARNING: auto should have value of bool type")

    def __repr__(self):
        return str(self.strategy)
