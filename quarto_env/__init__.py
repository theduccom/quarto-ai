from gymnasium.envs.registration import register

from quarto_env.quarto_env import QuartoEnv

register(
    id="QuartoEnv-v0",
    entry_point="quarto_env.quarto_env:QuartoEnv",
)

__all__ = ["QuartoEnv"]
