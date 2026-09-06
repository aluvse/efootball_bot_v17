from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ..core import Action, FootballAction, Movement


@dataclass
class Rollout:
    states: list = None
    movements: list = None
    football: list = None
    sprint: list = None
    rewards: list = None
    dones: list = None
    log_probs: list = None
    values: list = None

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.states = []
        self.movements = []
        self.football = []
        self.sprint = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def __len__(self):
        return len(self.states)


class FootballPolicy(nn.Module):
    def __init__(self, state_dim, hidden, movement_actions, football_actions, sprint_actions):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.movement = nn.Linear(hidden, movement_actions)
        self.football = nn.Linear(hidden, football_actions)
        self.sprint = nn.Linear(hidden, sprint_actions)
        self.value = nn.Linear(hidden, 1)
        self._init()

    def _init(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.constant_(self.value.bias, 0.0)
        nn.init.orthogonal_(self.movement.weight, gain=0.01)
        nn.init.orthogonal_(self.football.weight, gain=0.01)
        nn.init.orthogonal_(self.sprint.weight, gain=0.01)

    def forward(self, states):
        z = self.encoder(states)
        return self.movement(z), self.football(z), self.sprint(z), self.value(z).squeeze(-1)


def _gae(rewards, dones, values, last_value, gamma, lam):
    rewards = np.asarray(rewards, np.float32)
    dones = np.asarray(dones, np.float32)
    values = np.asarray(values + [last_value], np.float32)
    adv = np.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * non_terminal - values[t]
        gae = delta + gamma * lam * non_terminal * gae
        adv[t] = gae
    return adv, adv + values[:-1]


class PPOAgent:
    """Categorical PPO matching the structured football action contract."""

    def __init__(self, cfg, state_dim):
        self.cfg = cfg
        self.device = torch.device("cpu")
        self.net = FootballPolicy(
            state_dim,
            cfg.hidden,
            cfg.movement_actions,
            cfg.football_actions,
            cfg.sprint_actions,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=cfg.lr, eps=1e-5
        )
        self.update_count = 0
        self.bootstrap_done = False
        self.lock = threading.RLock()
        if cfg.resume:
            self.load(cfg.checkpoint)
        self.net.train()
        print(f"[PPO] device={self.device} updates={self.update_count}")

    @staticmethod
    def _mask_logits(movement_logits, football_logits, states):
        # Can't pass/shoot/cross when opponent has clear control.
        opp = states[:, 26 + 5] if states.shape[1] > 32 else torch.zeros(len(states), device=states.device)
        # V15 vector: ball 7 + active 7 + designated 7 + players 110 + possession one-hot 5.
        opp_control = states[:, 135] > 0.5 if states.shape[1] > 135 else torch.zeros(len(states), dtype=torch.bool, device=states.device)
        football_logits = football_logits.clone()
        football_logits[opp_control, int(FootballAction.PASS)] = -8.0
        football_logits[opp_control, int(FootballAction.CROSS)] = -8.0
        football_logits[opp_control, int(FootballAction.SHOOT)] = -8.0
        return movement_logits, football_logits

    def distribution(self, states):
        movement, football, sprint, value = self.net(states)
        movement, football = self._mask_logits(movement, football, states)
        return Categorical(logits=movement), Categorical(logits=football), Categorical(logits=sprint), value

    @torch.no_grad()
    def act(self, state, deterministic=False):
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with self.lock:
            d_m, d_f, d_s, value = self.distribution(x)
            if deterministic:
                m = torch.argmax(d_m.logits, dim=-1)
                f = torch.argmax(d_f.logits, dim=-1)
                s = torch.argmax(d_s.logits, dim=-1)
                logp = torch.zeros(1, device=self.device)
            else:
                m = d_m.sample(); f = d_f.sample(); s = d_s.sample()
                logp = d_m.log_prob(m) + d_f.log_prob(f) + d_s.log_prob(s)
            action = Action(int(m.item()), int(f.item()), int(s.item()))
            return action, float(logp.item()), float(value.item())

    def behavior_clone_step(self, batch_states, batch_actions):
        states = torch.as_tensor(batch_states, dtype=torch.float32, device=self.device)
        target_m = torch.as_tensor([a.movement for a in batch_actions], dtype=torch.long, device=self.device)
        target_f = torch.as_tensor([a.football for a in batch_actions], dtype=torch.long, device=self.device)
        target_s = torch.as_tensor([a.sprint for a in batch_actions], dtype=torch.long, device=self.device)
        with self.lock:
            m, f, s, _ = self.net(states)
            m, f = self._mask_logits(m, f, states)
            loss = F.cross_entropy(m, target_m) + F.cross_entropy(f, target_f) + 0.25 * F.cross_entropy(s, target_s)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()
        return float(loss.item())

    def update(self, rollout: Rollout, last_value: float):
        adv, returns = _gae(
            rollout.rewards, rollout.dones, rollout.values,
            last_value, self.cfg.gamma, self.cfg.gae_lambda,
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        states = torch.as_tensor(np.asarray(rollout.states), dtype=torch.float32, device=self.device)
        tm = torch.as_tensor(np.asarray(rollout.movements), dtype=torch.long, device=self.device)
        tf = torch.as_tensor(np.asarray(rollout.football), dtype=torch.long, device=self.device)
        ts = torch.as_tensor(np.asarray(rollout.sprint), dtype=torch.long, device=self.device)
        old_lp = torch.as_tensor(np.asarray(rollout.log_probs), dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)

        n = len(rollout)
        total = {"policy":0.0, "value":0.0, "entropy":0.0, "kl":0.0, "clip":0.0}
        batches = 0
        with self.lock:
            for _ in range(self.cfg.epochs):
                idxs = np.random.permutation(n)
                stop = False
                for start in range(0, n, self.cfg.minibatch):
                    idx = idxs[start:start+self.cfg.minibatch]
                    d_m, d_f, d_s, values = self.distribution(states[idx])
                    new_lp = d_m.log_prob(tm[idx]) + d_f.log_prob(tf[idx]) + d_s.log_prob(ts[idx])
                    entropy = d_m.entropy() + d_f.entropy() + d_s.entropy()
                    ratio = torch.exp((new_lp - old_lp[idx]).clamp(-20,20))
                    obj1 = ratio * adv_t[idx]
                    obj2 = torch.clamp(ratio, 1-self.cfg.clip, 1+self.cfg.clip) * adv_t[idx]
                    policy_loss = -torch.min(obj1, obj2).mean()
                    value_loss = F.mse_loss(values, returns_t[idx])
                    loss = policy_loss + self.cfg.value_coef*value_loss - self.cfg.entropy_coef*entropy.mean()
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                    self.optimizer.step()
                    with torch.no_grad():
                        kl = (old_lp[idx] - new_lp).mean().item()
                        clip_fraction = ((ratio-1.0).abs() > self.cfg.clip).float().mean().item()
                    total["policy"] += float(policy_loss.item())
                    total["value"] += float(value_loss.item())
                    total["entropy"] += float(entropy.mean().item())
                    total["kl"] += kl
                    total["clip"] += clip_fraction
                    batches += 1
                    if self.cfg.target_kl > 0 and kl > self.cfg.target_kl:
                        stop = True
                        break
                if stop:
                    break

            self.update_count += 1
        rollout.reset()
        return {k: v/max(1,batches) for k,v in total.items()} | {"update": self.update_count}

    def save(self, path: Path, bootstrap_done: Optional[bool] = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        if bootstrap_done is None:
            bootstrap_done = self.bootstrap_done
        with self.lock:
            torch.save({
                "format":"v24_structured_ppo",
                "model":self.net.state_dict(),
                "optimizer":self.optimizer.state_dict(),
                "update":self.update_count,
                "bootstrap_done":bool(bootstrap_done),
            }, tmp)
        self.bootstrap_done = bool(bootstrap_done)
        tmp.replace(path)
        print(f"[PPO] saved {path}")

    def load(self, path: Path):
        path = Path(path)
        if not path.exists():
            print("[PPO] no V24 checkpoint; starting new policy")
            return
        try:
            data = torch.load(path, map_location=self.device)
            if data.get("format") != "v24_structured_ppo":
                raise ValueError("incompatible checkpoint format")
            self.net.load_state_dict(data["model"])
            if "optimizer" in data:
                self.optimizer.load_state_dict(data["optimizer"])
            self.update_count = int(data.get("update",0))
            self.bootstrap_done = bool(data.get("bootstrap_done", False))
            print(f"[PPO] loaded {path} update={self.update_count} bootstrap_done={self.bootstrap_done}")
        except Exception as exc:
            print(f"[PPO] checkpoint ignored: {exc}")
