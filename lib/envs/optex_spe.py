"""SPE oracle loader for OptimalExecution, backed by pre-computed ``.npy``.

The ``.npy`` files (e.g. ``SPE_OptEx_5_0.019_4.npy``) produced by CumSPERL are
plain ``int64`` arrays of per-node actions. Indexing into them requires
reconstructing the exact BFS traversal of reachable (time, state) nodes used
at save time — the ``OptExTree`` routine below mirrors
``CumSPERL_ref/OptExTree.py`` but strips dependencies on the global ``settings``
module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def state_to_string(state) -> str:
    """Format a 3-vector state the same way CumSPERL's ``StateToString`` did,
    so that node keys match those the ``.npy`` was produced against."""
    parts = [f"{np.sign(x) * round(abs(float(x)), 6):.4f}" for x in state]
    return ", ".join(parts)


class _Node:
    __slots__ = ("height", "state", "child", "parent")

    def __init__(self, height, state):
        self.height = height
        self.state = np.asarray(state)
        self.child = []
        self.parent = {}  # parent_node -> list of (action, reward, prob)

    def update_parent(self, parent, info):
        if parent in self.parent:
            self.parent[parent].append(info)
            return False
        self.parent[parent] = [info]
        return True

    def __str__(self):
        return state_to_string(self.state)


class OptExTree:
    """Minimal BFS-built reachability tree keyed by (height, state_string)."""

    def __init__(self, env, initial_states, action_space_size):
        self.env = env
        self.action_space_size = action_space_size
        self.state_time_space = {}  # (height, str) -> Node (insertion-ordered)

        self.roots = [_Node(0, s) for s in initial_states]
        for root in self.roots:
            root.update_parent(None, (None, 0, 1.0 / len(initial_states)))
            self.state_time_space[(0, str(root))] = root

        self._build()

    def _build(self):
        last_layer = list(self.roots)
        layer_next = 0
        while last_layer:
            layer_next += 1
            new_layer = []
            for node in last_layer:
                cur_state = node.state
                for (action, w_bin) in [
                    (a, w) for a in range(self.action_space_size)
                    for w in range(self.env.num_w)
                ]:
                    new_state, reward, done = self.env.next_state(
                        cur_state, node.height, action, w_bin
                    )
                    prob = self.env._options_prob(w_bin)
                    new_node = _Node(node.height + 1, new_state)
                    new_node.update_parent(node, (action, reward, prob))

                    key = (new_node.height, str(new_node))
                    if key not in self.state_time_space:
                        self.state_time_space[key] = new_node
                        if not (done or layer_next == self.env.horizon):
                            new_layer.append(new_node)
                        node.child.append(new_node)
                    else:
                        existing = self.state_time_space[key]
                        if existing.update_parent(node, (action, reward, prob)):
                            node.child.append(existing)
            last_layer = new_layer


class SPEOracle:
    """Deterministic (time, state) -> action policy loaded from ``.npy``."""

    def __init__(self, env, spe_npy_path, initial_states=None):
        self.env = env
        self.action_list = np.load(str(spe_npy_path))
        if initial_states is None:
            initial_states = [env.reset()]
        self.tree = OptExTree(env, initial_states, env.action_space_size)

        # Mirror CumSPERL's ``NODE_DF``: columns [time, s0, s1, s2].
        rows = []
        for j, ((t, _), node) in enumerate(self.tree.state_time_space.items()):
            rows.append([t, float(node.state[0]), float(node.state[1]), float(node.state[2])])
        self.node_df = pd.DataFrame(rows, columns=["t", "s0", "s1", "s2"])

        if len(self.node_df) != len(self.action_list):
            raise RuntimeError(
                f"SPE npy length {len(self.action_list)} does not match "
                f"reconstructed tree size {len(self.node_df)} — env parameters "
                f"(horizon/sigma/num_w) likely disagree with the saved file."
            )

    def action_at(self, time, state) -> int:
        """Lookup SPE action, matching on (t, X_bin, logRet_bin) as in
        CumSPERL's ``GetTimeStateIndex`` for OptEx."""
        df = self.node_df[self.node_df["t"] == time]
        exact = df[(df["s2"] == float(state[2])) & (df["s0"] == float(state[0]))]
        if len(exact):
            idx = exact.index[0]
        else:
            df_x = df[df["s2"] == float(state[2])]
            if not len(df_x):
                return int(np.random.randint(self.env.action_space_size))
            diffs = (df_x["s0"] - float(state[0])).abs()
            idx = diffs.idxmin()
        return int(self.action_list[idx])

    def policy(self):
        """Callable ``policy(time, state) -> action``."""
        return lambda t, s: self.action_at(t, s)
