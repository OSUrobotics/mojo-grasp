from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from gym import Env
from stable_baselines3.common.vec_env import VecEnv
from typing import Any, ClassVar, Dict, Callable, Type, TypeVar, Tuple, Optional
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer, RolloutBuffer
import numpy as np
from gymnasium import spaces
from torch.nn import functional as F
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
import sys 
class PPOExpertData(PPO):
    def __init__(self, 
                 policy: str | type[ActorCriticPolicy], 
                 env: Env | VecEnv | str, 
                 expert_buffer: str | type[ReplayBuffer],
                 learning_rate: float | Callable[[float], float] = 0.0003, 
                 n_steps: int = 2048, 
                 batch_size: int = 64, 
                 n_epochs: int = 10, 
                 gamma: float = 0.99, 
                 gae_lambda: float = 0.95, 
                 clip_range: float | Callable[[float], float] = 0.2, 
                 clip_range_vf: None | float | Callable[[float], float] = None, 
                 normalize_advantage: bool = True, 
                 ent_coef: float = 0, 
                 vf_coef: float = 0.5, 
                 max_grad_norm: float = 0.5, 
                 use_sde: bool = False, 
                 sde_sample_freq: int = -1, 
                 rollout_buffer_class: type[RolloutBuffer] | None = None, 
                 rollout_buffer_kwargs: Dict[str, Any] | None = None, 
                 target_kl: float | None = None, 
                 stats_window_size: int = 100, 
                 tensorboard_log: str | None = None, 
                 policy_kwargs: Dict[str, Any] | None = None, 
                 verbose: int = 0, 
                 seed: int | None = None, 
                 device: th.device | str = "auto", 
                 _init_setup_model: bool = True):
        super().__init__(policy, env, tensorboard_log=tensorboard_log)
        self.expert_buffer = expert_buffer
        self.expert_coef=0.5*0.0001

    def evaluate_actions(self, obs, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy
    
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        expert_losses=[]
        continue_training = True
        # train for n_epochs epochs
        # print('starting the train cycle')
        # print(sys.getsizeof(self.rollout_buffer), sys.getsizeof(self.expert_buffer))
        expert_modifier = max(self.expert_coef*(1-2*self.num_timesteps/self._total_timesteps),0)
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # print('epoch', epoch, self.n_epochs)
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                # print('rollout size, expert size',sys.getsizeof(self.rollout_buffer), sys.getsizeof(self.expert_buffer))
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                expert_data = self.expert_buffer.sample(self.batch_size)
                # print(expert_data.observations.shape)
                # print('rollout data:', rollout_data.observations.shape)
                if len(expert_data.observations.shape) ==5:
                    t = expert_data.observations.squeeze(-1)
                    _, expert_log_prob, _ = self.policy.evaluate_actions(t, expert_data.actions)
                else:
                    _, expert_log_prob, _ = self.policy.evaluate_actions(expert_data.observations, expert_data.actions)
                expert_loss = -th.mean(expert_log_prob)
                # print(expert_loss,expert_loss.item(),expert_modifier)
                # print()
                expert_losses.append(expert_loss.item()* expert_modifier)
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + expert_loss * expert_modifier

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/expert_data_loss", np.mean(expert_losses)*expert_modifier)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
