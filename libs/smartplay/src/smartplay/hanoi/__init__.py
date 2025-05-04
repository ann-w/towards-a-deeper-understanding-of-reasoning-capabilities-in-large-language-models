from gym.envs.registration import register

from .hanoi_env import (Hanoi2Disk, Hanoi2DRewardShaping, Hanoi2DShowValid,
                        Hanoi2DShowValidRewardShaping, Hanoi3Disk,
                        Hanoi3DRewardShaping, Hanoi3DShowValid,
                        Hanoi3DShowValidRewardShaping, Hanoi4Disk)

environments = [
    ["Hanoi3Disk", "v0"],
    ["Hanoi4Disk", "v0"],
    # New environments
    ["Hanoi2Disk", "v0"],
    ["Hanoi2DRewardShaping", "v0"],
    ["Hanoi2DShowValid", "v0"],
    ["Hanoi2DShowValidRewardShaping", "v0"],
    ["Hanoi3DShowValid", "v0"],
    ["Hanoi3DRewardShaping", "v0"],
    ["Hanoi3DShowValidRewardShaping", "v0"],
]


for environment in environments:
    register(
        id="{}-{}".format(environment[0], environment[1]),
        entry_point="smartplay.hanoi:{}".format(environment[0]),
    )
