import argparse
from subprocess import call


def generate_rollouts(i):
    print("\n\nGenerating Rollouts\n\n")
    cmd = "python3 data/walker_generation_script.py --rollouts {} --rootdir {} --threads {} --iteration_num {}\n".format(
        int(rollouts_per_iteration), args.rollout_dir, int(num_threads), int(i)
    )
    print(cmd)
    call(cmd, shell=True)


def train_vae(i):
    print("\n\nTraining Vae\n\n")
    cmd = "python3 trainvae.py --logdir {} --iteration_num {} --dataset_dir {}\n".format(
        args.log_dir, int(i), args.rollout_dir
    )
    print(cmd)
    call(cmd, shell=True)


def train_mdrnn(i):
    print("\n\nTraining MDRNN\n\n")
    cmd = "python3 trainmdrnn.py --logdir {} --iteration_num {} --dataset_dir {} \n\n".format(
        args.log_dir, int(i), args.rollout_dir
    )
    print(cmd)
    call(cmd, shell=True)


def train_controller(i):
    print("\n\nTraining Controller\n\n")
    cmd = 'xvfb-run -s "-screen 0 1400x900x24" python3 traincontroller.py --logdir {} --n-samples 4 --pop-size 4 --target-return 950 --display --iteration_num {}\n\n'.format(
        args.log_dir, i
    )
    print(cmd)
    call(cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_iterations",
        type=int,
        help="Number of iterations of full traning of the world model, "
        "VAE, MDNRNN, C",
        default=10,
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        help="Dir to store all relevant model info, used in" "trainings scripts",
        default="exp_dir",
    )
    parser.add_argument(
        "--rollout_dir",
        type=str,
        help="Dir to store rollouts",
        default="datasets/walker",
    )
    args = parser.parse_args()

    total_rollouts = 1000
    assert total_rollouts % args.num_iterations == 0
    rollouts_per_iteration = total_rollouts / args.num_iterations
    num_threads = 10
    assert rollouts_per_iteration % num_threads == 0

    # loop n times
    for i in range(3, args.num_iterations):
        print("\n====================================\n")
        print(
            "Start iteration {} of iterative training the world" "model\n\n".format(i)
        )
        # generate rollouts
        generate_rollouts(i)

        # train vae
        train_vae(i)
        # train mdnrnn
        train_mdrnn(i)

        # train controller
        train_controller(i)
