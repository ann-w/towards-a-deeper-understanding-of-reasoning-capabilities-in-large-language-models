from gym.envs.registration import register

from .messenger_env import MessengerEnv

environments = [
    ["MessengerL1", "v0", {"lvl": 1}],
    ["MessengerL2", "v0", {"lvl": 2}],
    ["MessengerL3", "v0", {"lvl": 3}],
    ["MessengerL1Shaped", "v0", {"lvl": 1, "use_shaping": True}],
    ["MessengerL1NoRand", "v0", {"lvl": 1, "use_text_substitution": False}],
    [
        "MessengerL1ShapedNoRand",
        "v0",
        {"lvl": 1, "use_shaping": True, "use_text_substitution": False},
    ],
]

for env_name, version, kwargs in environments:
    register(
        id=f"{env_name}-{version}",
        entry_point="smartplay.messenger_emma:MessengerEnv",
        kwargs=kwargs,
    )
