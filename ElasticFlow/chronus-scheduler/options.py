import argparse
import ast

class Options:
    def __init__(self, ):
        parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
        parser.add_argument('--placement', default='random', choices=['random', 'consolidate', 'gandiva', 'local_search', 'consolidate_random', 'local_search_rev'], type=str, help='placement policy')
        parser.add_argument('--schedule', default='time-aware-with-lease', type=str, choices=['fifo', 'dlas', 'gittins', 'gandiva', 'time-aware', 'time-aware-with-lease', 'advanced-time-aware-with-lease', 'time-aware-with-lease-with-multi-tenancy', 'lease', 'fairness', 'stf', 'srf', 'ssf', 'srsf', 'srtf', 'tetri-sched', 'yarn-cs', 'themis', 'genie', 'sigma'], help='schedue policy')
        parser.add_argument('--cluster_partition', default='all', type=str, help='cluster partition way')
        parser.add_argument('--profile', default=False, type=ast.literal_eval, help='whether profile')
        parser.add_argument('--profile_node_num', default=4, type=int, help='node num')
        parser.add_argument('--profile_dry_run', default=False, type=ast.literal_eval, help='trace path')
        parser.add_argument('--profiler_duration', default=5, type=int, help='profile duration')
        parser.add_argument('--profile_method', default='no', type=str, help="log metrics destination")
        parser.add_argument('--submission_dist', default='no', type=str, help="submission_dist")
        parser.add_argument('--dynamic_profiler', default=False, type=ast.literal_eval, help='disable turn off operation')
        parser.add_argument('--num_switch', default=1, type=int, help='number of switch')
        parser.add_argument('--num_node_p_switch', default=75, type=int, help='number of num_node_p_switch')
        parser.add_argument('--num_gpu_p_node', default=8, type=int, help='number of num_gpu_p_node')
        parser.add_argument('--num_cpu_p_node', default=128, type=int, help='number of num_cpu_p_node')
        parser.add_argument('--mem_p_node', default=256, type=int, help='number of mem_p_node')
        parser.add_argument('--check_time_interval', default=1, type=int, help='number of mem_p_node')
        parser.add_argument('--trace', default='data/trace_job.csv', type=str, help='trace path')
        parser.add_argument('--user_partition', default=None, type=str, help='user partition')
        parser.add_argument('--user_partition_size', default=None, type=str, help='user partition size')
        parser.add_argument('--save_log_dir', default='results/', type=str, help='log path')
        parser.add_argument('--ident', default='', type=str, help='identifier')
        parser.add_argument('--aggressive', default=False, type=ast.literal_eval, help='aggressive')
        parser.add_argument('--disable_turn_off', default=False, type=ast.literal_eval, help='disable turn off operation')
        parser.add_argument('--disable_force_guarantee', default=False, type=ast.literal_eval, help='disable turn off operation')
        parser.add_argument('--disable_noise_tolerance', default=False, type=ast.literal_eval, help='disable noise tolerance')
        parser.add_argument('--noise_diff', default=0, type=float, help='noise diff')
        parser.add_argument('--mip_objective', default='minimize', type=str, help='mip solver objective function')
        parser.add_argument('--lease_term_interval', default=60, type=int, help='lease specific: lease term interval')
        parser.add_argument('--disc_priority_k', default=5, type=int,
                            help='lease specific: how many discrete priority class')
        parser.add_argument('--job_selection', default='random',
                            choices=['random', 'fifo', 'smallestfirst', '2das', 'shortestremainfirst', 'fairness', 'disc_fairness', 'job_reward', 'genie'],
                            type=str, help='job selection policy')
        parser.add_argument('--log_level', default='INFO', choices=['INFO', 'DEBUG', 'ERROR'],
                            type=str, help='log level')
        parser.add_argument('--share', default='data/share.csv', type=str, help='user share path')
        parser.add_argument('--name_list', default='data/name.lst', type=str, help='user name list')
        parser.add_argument('--fairness_output', default='fairness.csv', type=str, help='fairness.csv')
        parser.add_argument('--numgpu_fallback_threshold', default=5, type=int, help='numgpu fallback length threshold')
        parser.add_argument('--dist_trace_path', default='', type=str, help='distribution file path')
        parser.add_argument('--metrics', default=False, type=bool, help="log metrics")
        parser.add_argument('--metrics_path', default='metrics.csv', type=str, help="log metrics destination")
        parser.add_argument('--simulation', default=False, type=bool, help="simulation or real cluster experiment")
        parser.add_argument('--gpu_type', default='A100', type=str, help="gpu type")
        
        self.args = parser.parse_args()
    
    def init(self, ):
        return self.args


Singleton = Options()


_allowed_symbols = [
    'Singleton'
]
