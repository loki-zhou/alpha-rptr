from ray.rllib.algorithms.callbacks import DefaultCallbacks

class TradeMetricsCallbacks(DefaultCallbacks):
    """LeelaChessZero callbacks.
    If you use custom callbacks, you must extend this class and call super()
    for on_episode_start.
    """

    def on_episode_start(
        self,
        *,
        worker,
        base_env,
        policies,
        episode,
        env_index,
        **kwargs
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        episode.custom_metrics["portfolio_return"] = 0
        episode.custom_metrics["position_changes"] = 0
        episode.custom_metrics["max_drawdown"] = 0


    def on_episode_end(
        self,
        *,
        worker,
        base_env,
        policies,
        episode,
        env_index,
        **kwargs
    ):
        env = base_env.get_sub_environments()[0]
        metrics = env.get_metrics()
        episode.custom_metrics["portfolio_return"] = (float(metrics["Portfolio Return"][:-1]))
        episode.custom_metrics["position_changes"] = (float(metrics["Position Changes"][:-1]))
        episode.custom_metrics["max_drawdown"] = (float(metrics["Max Drawdown"][:-1]))