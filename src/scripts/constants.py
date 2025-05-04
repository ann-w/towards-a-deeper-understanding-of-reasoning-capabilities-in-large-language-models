# These numbers are based on the number of samples used in the paper
ENV_TRAJECTORY_SAMPLES = {
    "BanditTwoArmedHighLowFixed": 50,
    "RockPaperScissorBasic": 50,
    "Hanoi3Disk": 30,
    "MessengerL1": 10,  # used to be 2, set on 10
    "MessengerL2": 5,
    "Crafter": 2,
    # Add new environments Hanoi
    "Hanoi2Disk": 30,
    "Hanoi2DShowValid": 30,
    "Hanoi2DRewardShaping": 30,
    "Hanoi2DShowValidRewardShaping": 30,
    "Hanoi3DShowValid": 30,
    "Hanoi3DRewardShaping": 30,
    "Hanoi3DShowValidRewardShaping": 30,
    # Add new environments Messenger
    "MessengerL1Shaped": 10,
    "MessengerL1NoRand": 10,
    "MessengerL1ShapedNoRand": 10,
}
