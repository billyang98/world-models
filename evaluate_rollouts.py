""" Test controller """
import argparse
from os.path import join, exists
from os import makedirs
from utils.misc import RolloutGenerator
import torch
import shutil
import heapq

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, help='Where models are stored.')
parser.add_argument('--iteration_num', type=int,
                    help="Iteration number of which controller to use")
parser.add_argument('--rollouts', type=int, 
                    help='Number of rollouts to generate',
                    default=100)
parser.add_argument('--rollouts_dir', type=str, 
                    help='Directory to store the rollouts')
parser.add_argument('--video_dir', type=str, 
                    help='Directory to store the videos',
                    default="videos")
parser.add_argument('--do_not_store_videos', type=bool, 
                    help='Boolean to not store videos or store them',
                    default=False)
args = parser.parse_args()

ctrl_file = join(args.logdir, 'ctrl', 'best.tar')
if args.iteration_num is not None:
    ctrl_file = join(args.logdir, 'ctrl', 'iter_{}'.format(args.iteration_num),
                    'best.tar')

assert exists(ctrl_file),\
    "Controller was not trained..."

device = torch.device('cpu')

generator = RolloutGenerator(args.logdir, device, 2000, args.iteration_num)

total_reward = 0
top5rollouts = []
video_dir = args.video_dir
if args.iteration_num is not None:
    video_dir = join(args.video_dir, "iter_{}".format(args.iteration_num))
makedirs(video_dir, exist_ok=True)

f = open(join(video_dir, "rewards"),"w+")
f.write("Reward Rollout#\n")

for i in range(0, args.rollouts):
  with torch.no_grad():
      store_video_dir = None if args.do_not_store_videos else video_dir
      reward = -generator.rollout(None, rollout_dir=args.rollouts_dir, rollout_num=i, video_dir=store_video_dir)
      print("Reward roll_{} = {}".format(i, reward))
      f.write("{} roll_{}\n".format(reward, i))
      total_reward += reward

print("Total Reward = {}".format(total_reward))
print("Num Rollouts = {}".format(args.rollouts))
print("Average Reward = {}".format(total_reward / args.rollouts))
f.write("\n\n {} Total Reward\n".format(total_reward))
f.write("{} Num Rollouts\n".format(args.rollouts))
f.write("{} Average Reward\n".format(total_reward / args.rollouts))
f.close()


