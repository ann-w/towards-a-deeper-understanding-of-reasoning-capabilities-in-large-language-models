from gym.envs.registration import register

from .bandit import (BanditTwoArmedDeterministicFixed,
                     BanditTwoArmedHighHighFixed, BanditTwoArmedHighLowFixed,
                     BanditTwoArmedLowLowFixed)

environments = [
    ["BanditTwoArmedDeterministicFixed", "v0"],
    ["BanditTwoArmedHighHighFixed", "v0"],
    ["BanditTwoArmedHighLowFixed", "v0"],
    ["BanditTwoArmedLowLowFixed", "v0"],
]

for environment in environments:
    register(
        id="{}-{}".format(environment[0], environment[1]),
        entry_point="smartplay.bandits:{}".format(environment[0]),
    )
