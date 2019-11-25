"""
Encapsulate generate data to make it parallel
"""
from os import makedirs
from os.path import join
import argparse
from multiprocessing import Pool
from subprocess import call

parser = argparse.ArgumentParser()
parser.add_argument('--rollouts', type=int, help="Total number of rollouts.")
parser.add_argument('--threads', type=int, help="Number of threads")
parser.add_argument('--rootdir', type=str, help="Directory to store rollout "
                    "directories of each thread")
parser.add_argument('--policy', type=str, choices=['brown', 'white'],
                    help="Directory to store rollout directories of each thread",
                    default='brown')
parser.add_argument('--iteration_num', type=int,
                    help="Iteration number of full traning of the world model, "
                    "VAE, MDNRNN, C")
args = parser.parse_args()

assert args.rollouts % args.threads == 0
rpt = args.rollouts // args.threads

def _threaded_generation(i):
    tdir = join(args.rootdir, 'thread_{}'.format(i))
    if args.iteration_num is not None:
      tdir = join(args.rootdir, 'iter_{}'.format(args.iteration_num),'thread_{}'.format(i))
    makedirs(tdir, exist_ok=True)
    cmd = ['xvfb-run', '-s', '"-screen 0 1400x900x24"']
    cmd += ['--server-num={}'.format(i + 1)]
    if args.iteration_num is not None and args.iteration_num > 0:
        print("\nGenerating rollouts from controller")
        # do the controller rollout 
        cmd += ["python3 test_controller.py --logdir exp_dir --rollouts {} --rollouts_dir {} --iteration_num {}".format(rpt, tdir, args.iteration_num-1)
    else:
        print("\nGenerating random rollouts")
        cmd += ["python3", "-m", "data.walker", "--dir",
                tdir, "--rollouts", str(rpt), "--policy", args.policy]
        if args.iteration_num is not None:
          cmd += ["--iteration_num",str(args.iteration_num)]
    cmd = " ".join(cmd)
    print(cmd)
    call(cmd, shell=True)
    return True


with Pool(args.threads) as p:
    p.map(_threaded_generation, range(args.threads))
