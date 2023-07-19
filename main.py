import scenario
import utils

def main():
    args = utils.get_args()
    print(args)
    method = getattr(scenario, args.scenario)
    method(args)


if __name__ == "__main__":
    main()