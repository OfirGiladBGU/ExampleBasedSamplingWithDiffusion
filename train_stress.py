from utils.Config import ParseTrainConfig, parse_eval
import argparse


if __name__ == "__main__":
    # CONFIG_PATH = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/config/GBN_stress/config1.json"
    # CONFIG_PATH = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/config/GBN_stress/config2.json"
    # CONFIG_PATH = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/config/GBN_stress/config2_V2.json"

    # FM training
    CONFIG_PATH = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/config/GBN/config_fm.json"
    ITS = int(1e10)
    TRAINING_TIME = 180  # in minutes
    # TRAINING_TIME = 360  # in minutes

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-c", "--config", help="Path to config file", required=False, type=str, default=CONFIG_PATH)
    parser.add_argument("--its", help="Number of optimization steps", required=False, type=int, default=ITS)
    parser.add_argument("-t", "--time", help="Training time in minutes", required=False, type=int, default=TRAINING_TIME,)
    parser.add_argument("--tqdm", help="Adds progress bar for training", required=False, type=bool, default=True)

    args = parser.parse_args()

    trainer = ParseTrainConfig(args.config)

    # Lowest of both 'args.its, args.time' will be used
    trainer.train(args.its, args.tqdm, args.time)