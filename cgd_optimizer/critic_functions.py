import torch

def get_advantage(val, rew, gamma = 0.99, tau = 0.95):
    advantage = torch.zeros(rew.shape[0])
    for i in reversed(range(rew.shape[0])):
        delta = rew[i] + gamma * val[i+1] - val[i]
        try:
            advantage[i] = delta + gamma * tau * advantage[i+1]
        except:
            advantage[i] = delta
    return advantage

def critic_update(state_mb, return_mb, q, optim_q):
    val_loc = q(state_mb)
    critic_loss = (return_mb - val_loc).pow(2).mean()

    optim_q.zero_grad()
    critic_loss.backward()
    optim_q.step()

    del val_loc
    critic_loss_numpy = critic_loss.detach().cpu().numpy()
    del critic_loss

    yield critic_loss_numpy, 1