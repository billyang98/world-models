""" Test controller """
import argparse
from os.path import join, exists
from utils.misc import RolloutGenerator
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, help='Where models are stored.')
parser.add_argument('--iteration_num', type=int,
                    help="Iteration number of which controller to use")
parser.add_argument('--rollouts', type=int, 
                    help='Number of rollouts to generate',
                    default=1)
parser.add_argument('--rollouts_dir', type=str, 
                    help='Directory to store the rollouts')
args = parser.parse_args()

ctrl_file = join(args.logdir, 'ctrl', 'best.tar')
if args.iteration_num is not None:
    ctrl_file = join(args.logdir, 'ctrl', 'iter_{}'.format(args.iteration_num),
                    'best.tar')

assert exists(ctrl_file),\
    "Controller was not trained..."

device = torch.device('cpu')

generator = RolloutGenerator(args.logdir, device, 1000, args.iteration_num)

for i in range(0, args.rollouts):
  with torch.no_grad():
      generator.rollout(None, rollout_dir=args.rollouts_dir, rollout_num=i)
