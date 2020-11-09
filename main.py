import argparse
from model import train_test_model


def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Choose model:"
        " resnet50,"
        " densenet201,"
        " inception_resnet_v2,"
        " resnext50_32x4d,"
        " resnext101_32x8d,"
        " efficientnet_b4",
    )

    parser.add_argument(
        "--lr", "-l", default=1e-3, type=float, help="Base learning rate"
    )

    parser.add_argument(
        "--epochs", "-e", default=50, type=int, help="Number of epochs"
    )

    parser.add_argument(
        "--e_name",
        "-e_n",
        type=str,
        default="DL_model",
        help="Extra model's name avoid to replace the same name of model",
    )

    return parser.parse_args()


# choose model to train and test data
if __name__ == "__main__":
    args = process_command()
    if str(args.model) in [
        "resnet50",
        "densenet201",
        "inception_resnet_v2",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "efficientnet_b4",
    ]:
        train_test_model(args.model, args.lr, args.epochs, args.e_name)
    else:
        print(
            "Input error, wrong model name or multiple models(Only one model can trained)"
        )
