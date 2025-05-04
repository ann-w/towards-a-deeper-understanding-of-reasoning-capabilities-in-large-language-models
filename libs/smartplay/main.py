import argparse
import os

os.environ["MINEDOJO_HEADLESS"] = "1"
import smartplay

from scripts.play_game import run


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm_name", type=str, default="microsoft/phi-2", help="Name of the LLM"
    )
    parser.add_argument(
        "--env_names",
        type=str,
        default="MessengerL1,MessengerL2,MessengerL3,RockPaperScissorBasic,RockPaperScissorDifferentScore,BanditTwoArmedDeterministicFixed,BanditTwoArmedHighHighFixed,BanditTwoArmedHighLowFixed,BanditTwoArmedLowLowFixed,Hanoi3Disk,Hanoi4Disk,MinedojoCreative0,MinedojoCreative1,MinedojoCreative2,MinedojoCreative4,MinedojoCreative5,MinedojoCreative7,MinedojoCreative8,MinedojoCreative9,Crafter",
        help="Comma separated list of environments to run",
    )
    parser.add_argument(
        "--env_steps",
        type=int,
        default=None,
        help="Number of steps to run the environment for. If not provided, the default number of steps for the environment will be used.",
    )
    parser.add_argument(
        "--env_iter",
        type=int,
        default=1,
        help="Number of iterations to run the environment for. If not provided, the default number of iterations for the environment will be used.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.env_names is None:
        args.env_names = ",".join(smartplay.benchmark_games_v0)

    score_dict = {}

    for env_name in args.env_names.split(","):
        score_dict[env_name] = run(
            env_name, args.llm_name, args.env_steps, args.env_iter
        )

    print("Normalized scores on each task:", score_dict)
    print("Capability scores of the LLM:", smartplay.analyze_capabilities(score_dict))


if __name__ == "__main__":
    main()
