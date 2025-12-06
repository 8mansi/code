import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import numpy as np

# PPO NETWORK
class PPOActorCritic(nn.Module):
    def __init__(self, state_dim=9, action_dim=4): 
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.policy = nn.Linear(128, action_dim)
        self.value = nn.Linear(128, 1)

        for m in self.shared.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.zeros_(m.bias)

        for m in self.policy.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.zeros_(m.bias)

        for m in self.value.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.zeros_(m.bias)

        # Orthogonal initialization for stability
        for m in self.shared:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.constant_(self.policy.bias, 0.0)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.constant_(self.value.bias, 0.0)

    def forward(self, x):
        x = self.shared(x)
        return self.policy(x), self.value(x)


# PPO AGENT
class PPOAgent:
    def __init__(
        self,
        state_dim=9,
        action_dim=4,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99,
        clip_eps=0.2,
        update_epochs=8,
        batch_size=64,
        min_batch_size=256,
        entropy_coef=0.03
    ):
        self.model = PPOActorCritic(state_dim=state_dim, action_dim=action_dim)
        self.optimizer = optim.Adam([
            {"params": self.model.policy.parameters(), "lr": actor_lr},
            {"params": self.model.value.parameters(), "lr": critic_lr},
            {"params": self.model.shared.parameters(), "lr": actor_lr}  # usually shared with actor
        ])


        self.gamma = gamma
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        self.entropy_coef = entropy_coef
        self.last_action = None
        self.memory = {"states": [], "actions": [], "logprobs": [], "rewards": [], "dones": []}

    # ---------- action selection ----------
    def act(self, state):
        # state: numpy array or list (state_dim,)
        state_t = torch.FloatTensor(state).unsqueeze(0)  # make batch dim: (1, state_dim)
        logits, _ = self.model(state_t)                  # logits shape: (1, action_dim)
        logits = logits.squeeze(0)                       # shape: (action_dim,)

        safe_actions = [0, 1, 2, 3]  # all actions
        front_dist = state[4]  # front distance
        left_dist  = state[0]  # left distance
        right_dist = state[1]  # right distance
        # print("#################### front dist", front_dist, " left_dist ", left_dist, " right dist ", right_dist)

        if front_dist > 1 - 0.35 and 0 in safe_actions:
            safe_actions.remove(0)
        if left_dist > 1 - 0.25 and 1 in safe_actions:
            safe_actions.remove(1)
        if right_dist > 1 - 0.25 and 2 in safe_actions:
            safe_actions.remove(2)
        if front_dist < 0.4 and 3 in safe_actions:
            safe_actions.remove(3)

        # Mask logits
        mask = torch.zeros_like(logits)
        mask[safe_actions] = 1.0
        masked_logits = logits + (mask + 1e-8).log()  # avoid log(0)

        dist_pi = dist.Categorical(logits=masked_logits)
        action = dist_pi.sample()
        logprob = dist_pi.log_prob(action)

        # Return plain python scalars (safe to store)
        return int(action.item()), float(logprob.item())

    def remember(self, s, a, lp, r, d):
        if self.last_action is not None and a == 3 and self.last_action == 3:
            r -= 0.02
        self.memory["states"].append(np.array(s, dtype=np.float32))
        self.memory["actions"].append(int(a))
        self.memory["logprobs"].append(float(lp))
        self.memory["rewards"].append(float(r))
        self.memory["dones"].append(bool(d))
        self.last_action = a

    # ---------- PPO update ----------
    def train(self, force=False, entropy_coef=None):
        """
        Runs PPO update. If not enough samples (len(states) < min_batch_size) and not force,
        update is skipped to avoid unstable small-batch updates.
        """
        if entropy_coef is None:
            entropy_coef = self.entropy_coef

        mem_size = len(self.memory["states"])
        if mem_size == 0:
            print("PPO.train(): memory empty -> nothing to do.")
            return 0.0

        if (not force) and mem_size < self.min_batch_size:
            print(f"PPO.train(): insufficient samples ({mem_size} < {self.min_batch_size}) -> skipping update.")
            return 0.0

        print(f"PPO.train(): Starting update on {mem_size} samples.")

        # Convert memory to tensors
        states = torch.FloatTensor(np.array(self.memory["states"], dtype=np.float32))  # (N, S)
        actions = torch.LongTensor(self.memory["actions"])                            # (N,)
        old_logprobs = torch.FloatTensor(self.memory["logprobs"])                    # (N,)

        # compute returns (discounted)
        returns = []
        G = 0.0
        for r, d in zip(reversed(self.memory["rewards"]), reversed(self.memory["dones"])):
            if d:
                G = 0.0
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)  # (N,)
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

        # Basic NaN check
        if torch.isnan(states).any() or torch.isnan(returns).any():
            print("PPO.train(): NaN detected in states/returns — clearing memory and aborting update.")
            self.memory = {k: [] for k in self.memory}
            return 0.0

        dataset = torch.utils.data.TensorDataset(states, actions, old_logprobs, returns)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        final_loss = 0.0
        for _epoch in range(self.update_epochs):
            for (s_batch, a_batch, old_lp_batch, ret_batch) in loader:
                # Evaluate current policy
                logits, values = self.model(s_batch)  # logits: (B, A), values: (B,1)
                if torch.isnan(logits).any() or torch.isnan(values).any():
                    print("PPO.train(): NaN in model outputs — aborting this update epoch.")
                    continue

                dist_pi = dist.Categorical(logits=logits)
                new_logprobs = dist_pi.log_prob(a_batch)  # (B,)
                entropy = dist_pi.entropy().mean()        # scalar

                # Compute advantages
                values = values.squeeze(1)                 # (B,)
                advantages = ret_batch - values            # (B,)
                critic_loss = 0.5 * (advantages.pow(2).mean())  

                # Robust normalization: avoid divide-by-zero
                adv_mean = advantages.mean()
                adv_std = advantages.std(unbiased=False)   # population std
                if torch.isnan(adv_std) or adv_std.item() < 1e-8:
                    # If std is too small, just center to zero-mean and skip scaling
                    advantages = advantages - adv_mean
                else:
                    advantages = (advantages - adv_mean) / (adv_std + 1e-8)

                # PPO surrogate
                ratio = torch.exp(new_logprobs - old_lp_batch)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages

                actor_loss = -torch.min(surr1, surr2).mean()

                loss = actor_loss + critic_loss - self.entropy_coef * entropy

                # Safe gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                final_loss = float(loss.detach().cpu().item())

        # Clear memory after update
        self.memory = {k: [] for k in self.memory}
        print("PPO.train(): Update finished. Final loss:", final_loss)
        

        return final_loss, actor_loss, critic_loss, entropy

    def save_model(self, filepath):
        """Saves the state dictionary of the underlying ActorCritic network."""
        torch.save(self.model.state_dict(), filepath)
        print(f"PPOAgent: Model saved successfully to {filepath}")

    def load_model(self, filepath):
        """Loads the state dictionary into the underlying ActorCritic network."""
        try:
            self.model.load_state_dict(torch.load(filepath))
            print(f"PPOAgent: Model loaded successfully from {filepath}")
            self.model.eval() # Important for inference
        except FileNotFoundError:
            print(f"PPOAgent: Error - Model file not found at {filepath}. Starting from scratch.")
        except RuntimeError as e:
            print(f"PPOAgent: Error loading model state dict. Error: {e}")