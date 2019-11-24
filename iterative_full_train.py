import argparse
from subprocess import call

def generate_random_rollouts():
    print("\n\nGenerating Random Rollouts\n\n")
    cmd ="python3 data/walker_generation_script.py --rollouts {} --rootdir {} --threads {} --iteration_num {}\n".format(int(rollouts_per_iteration), args.rollout_dir, int(num_threads), int(i))
    print(cmd)
    call(cmd, shell=True)

def train_vae():
    # TODO: don't hardcode epochs
    print("\n\nTraining Vae\n\n")
    cmd = "python3 trainvae.py --epochs 1 --logdir {} --iteration_num {} --dataset_dir {}\n".format(args.log_dir, int(i), args.rollout_dir)
#    call("python3 trainvae.py --logdir {} --iteration_num {} --dataset_dir "
#    "{}".format(args.log_dir, i, args.rollout_dir)) 
    print(cmd)
    call(cmd, shell=True)

def train_mdrnn():
    # TODO: don't hardcode epochs
    print("\n\nTraining MDRNN\n\n")
#    call("python3 trainmdrnn.py --logdir {} --iteration_num {} --dataset_dir "
#    "{}".format(args.log_dir, i, args.rollout_dir))
    cmd = "python3 trainmdrnn.py --epochs 1 --logdir {} --iteration_num {} --dataset_dir {} \n\n".format(args.log_dir, int(i), args.rollout_dir)
    print(cmd)
    call(cmd, shell=True)

def train_controller():
    print("\n\nTraining Controller\n\n")
    cmd = "python3 traincontroller.py --logdir {} --n-samples 4 --pop-size 4 --target-return 950 --display --iteration_num {}\n\n".format(args.log_dir, i)
    print(cmd)
    call(cmd, shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iterations', type=int,
                        help="Number of iterations of full traning of the world model, "
                        "VAE, MDNRNN, C", default=10)
    parser.add_argument('--log_dir', type=str,
                        help="Dir to store all relevant model info, used in"
                        "trainings scripts", default="exp_dir")
    parser.add_argument('--rollout_dir', type=str,
                        help="Dir to store rollouts", default="datasets/walker")
    args = parser.parse_args()

    #total_rollouts = 1000
    total_rollouts = 100 #TODO: change back to 1000
    assert total_rollouts % args.num_iterations == 0
    rollouts_per_iteration = total_rollouts / args.num_iterations
    num_threads = 10
    assert rollouts_per_iteration % num_threads == 0

    # loop n times
    for i in range(0, 2):
        print("\n====================================\n")
        print("Start iteration {} of iterative training the world"
        "model\n\n".format(i))
        # generate rollouts
        if i == 0:
            # random if first time
            generate_random_rollouts()
        else: 
            # using some policy if not
            # TODO: generate rollouts using model from last iteration
            generate_random_rollouts()

        # train vae
        train_vae()
        # train mdnrnn
        train_mdrnn()

        # train controller
        train_controller()
     

