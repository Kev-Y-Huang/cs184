import gym
from learner import *
from main import get_args

def test(args):
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    if args.env in DISCRETE_ENVS:
        action_dim = env.action_space.n
    else:
        action_dim = env.action_space.shape[0]
    if args.dagger:
        policy_save_path = os.path.join(args.policy_save_dir, 'dagger' ,f'{args.env}.pt')
        model = DAGGER(state_dim, action_dim, args)
        model.load(policy_save_path)
        
    else:
        policy_save_path = os.path.join(args.policy_save_dir, 'bc', f'{args.env}.pt')
        model = BC(state_dim, action_dim, args)
        model.load(policy_save_path)
    lengths = []
    rewards = []
    for _ in range(10):
        done = False
        ob = env.reset()

        length = 0
        reward = 0

        while not done:
            env.render()
            qs = model.get_logits(torch.from_numpy(ob).float()) 
            a = qs.argmax().numpy()

            next_ob, r, done, _ = env.step(a)
            ob = next_ob
            length += 1
            reward += r

        env.close()
        lengths.append(length)
        rewards.append(reward)

    print(f'average episode length: {np.mean(lengths)}')
    print(f'average reward incurred: {np.mean(rewards)}')

if __name__ == '__main__':
    args = get_args()
    test(args)