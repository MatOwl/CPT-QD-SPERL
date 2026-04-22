"""Optimal execution environment, migrated from CumSPERL_ref/environments.py.

Cleaned of dependencies on the global ``settings`` module: the action-space
size is now a constructor argument. Observation is a 3D integer tuple
(log-return bin, remaining-trade bin, holding bin), suitable for tabular FA.
"""

import numpy as np
from scipy.stats import norm
import gym
from gym import spaces


class OptimalExecution(gym.Env):
    metadata = {"render.modes": ["console"]}

    total_holding = 10 ** 6
    initial_stock_price = 50

    n_bins_price = 200
    discretise_factor_price = 1
    n_bins_N = 10
    n_bins_X = 10

    def __init__(
        self,
        horizon=5,
        sigma=0.019,
        num_w=4,
        action_space_size=11,
        gamma=25,
        eta=25,
        epsilon=0.0625,
    ):
        super().__init__()

        self.horizon = horizon
        self.T = horizon
        self.N = horizon

        self.sigma = sigma
        self.num_w = num_w

        self.action_space_size = action_space_size
        self.gamma = gamma * 10 ** (-8)
        self.eta = eta * 10 ** (-7)
        self.epsilon = epsilon
        self.lamb = 10 ** (-6)

        self.action_space = spaces.Discrete(action_space_size)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, 0, 0]),
            high=np.array([np.inf, 1.0, 1.0]),
            dtype=np.float32,
        )

        self.options_list = [
            (i, j) for i in range(action_space_size) for j in range(num_w)
        ]

        self.tau = self.T / self.N
        self.remaining_num_trade = self.N
        self.holding = self.total_holding
        self.stock_price = self.initial_stock_price
        self.log_return = 0.0
        self.financial_performance = 0.0
        self.time_step = 0

        # Barberis-style alias for downstream code expecting ``init_wealth``.
        self.init_wealth = self.total_holding

    # ---------- gym API ----------
    def reset(self, seed=None, init_state=None, **kwargs):
        """Standard reset to s_0. If ``init_state`` is provided as the discrete
        3-vector (logRet_bin, remain_bin, X_bin), invert the discretisation to
        restore the underlying continuous state — used for per-state V^π
        rollouts in paper-style evaluation."""
        if seed is not None:
            np.random.seed(seed)
        if init_state is None:
            self.time_step = 0
            self.remaining_num_trade = self.N
            self.holding = self.total_holding
            self.stock_price = self.initial_stock_price
            self.log_return = 0.0
        else:
            p_bin, n_bin, x_bin = (float(v) for v in init_state)
            self.remaining_num_trade = n_bin / self.n_bins_N * self.N
            self.holding = x_bin / self.n_bins_X * self.total_holding
            self.log_return = p_bin / (
                self.discretise_factor_price * self.n_bins_price
            )
            self.stock_price = self.initial_stock_price * np.exp(self.log_return)
            self.time_step = int(self.N - self.remaining_num_trade)
        self.financial_performance = 0.0
        return self._get_state()

    def step(self, action, debug=False):
        # Guard against stepping past the terminal state — happens when a
        # rollout is launched from a t=T leaf node in paper-style evaluation.
        if self.remaining_num_trade <= 0:
            return self._get_state(), 0.0, True, {}

        w_probs = np.array([self._options_prob(i) for i in range(self.num_w)])
        w_probs /= w_probs.sum()
        w = self._compute_w(np.random.choice(np.arange(self.num_w), p=w_probs))

        if self.remaining_num_trade == 1:
            shares_sold = self.holding
        else:
            shares_sold = self.holding * action / (self.action_space_size - 1)

        price_per_share = self.stock_price - self._h(shares_sold / self.tau)
        self.holding -= shares_sold

        self.stock_price += self.sigma * self.tau ** 0.5 * w * self.stock_price
        self.stock_price -= self.tau * self._g(shares_sold / self.tau)

        self.log_return = np.log(self.stock_price / self.initial_stock_price)
        self.remaining_num_trade -= 1
        self.time_step += 1

        reward = (
            (price_per_share - self.initial_stock_price)
            * shares_sold
            / self.initial_stock_price
            / self.total_holding
        )
        self.financial_performance += price_per_share * shares_sold

        done = self.remaining_num_trade == 0
        return self._get_state(), reward, done, {}

    def render(self, mode="console"):
        if mode != "console":
            raise NotImplementedError
        print(
            f"t={self.time_step}, remain={self.remaining_num_trade}, "
            f"hold={self.holding}, price={self.stock_price}"
        )

    def close(self):
        pass

    # ---------- helpers ----------
    def next_state(self, state, time, action, w_bin):
        """Deterministic transition used by the SPE tree builder.

        ``state`` is a 3-vector in the same encoding as ``_get_state``;
        ``w_bin`` is the discrete market-shock index in [0, num_w).
        """
        w = self._compute_w(w_bin)
        remaining = state[1] / self.n_bins_N * self.N
        holding = state[2] / self.n_bins_X * self.total_holding
        stock_price = (
            np.exp(state[0] / self.n_bins_price / self.discretise_factor_price)
            * self.initial_stock_price
        )

        if remaining == 1:
            shares_sold = holding
        else:
            shares_sold = holding * action / (self.action_space_size - 1)

        price_per_share = stock_price - self._h(shares_sold / self.tau)
        holding -= shares_sold
        stock_price += self.sigma * self.tau ** 0.5 * w * stock_price
        stock_price -= self.tau * self._g(shares_sold / self.tau)

        log_return = np.log(stock_price / self.initial_stock_price)
        remaining -= 1

        reward = (
            (price_per_share - self.initial_stock_price)
            * shares_sold
            / self.initial_stock_price
            / self.total_holding
        )
        done = remaining == 0
        output = np.array(
            [
                log_return * self.n_bins_price * self.discretise_factor_price // 1,
                remaining / self.N * self.n_bins_N // 1,
                holding / self.total_holding * self.n_bins_X // 1,
            ],
            dtype=np.intc,
        )
        return output, reward, done

    def _get_state(self):
        return np.array(
            [
                self.log_return
                * self.discretise_factor_price
                * self.n_bins_price
                // 1,
                self.remaining_num_trade / self.N * self.n_bins_N // 1,
                self.holding / self.total_holding * self.n_bins_X // 1,
            ],
            dtype=np.intc,
        )

    def _compute_w(self, w):
        ls = np.linspace(-4, 4, self.num_w + 1)
        return ls[w] + (ls[w + 1] - ls[w]) / 2

    def _options_prob(self, w):
        ls = np.linspace(-4, 4, self.num_w + 1)
        return norm.cdf(ls[w + 1]) - norm.cdf(ls[w])

    def _g(self, v):
        return self.gamma * v

    def _h(self, v):
        return self.epsilon * np.sign(v) + self.eta * v
