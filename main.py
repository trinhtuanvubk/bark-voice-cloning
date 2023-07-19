import scenario
from utils import get_args

def main():
    args = get_args()
    method = getattr(scenario, args.scenario)
    method(args)


if __name__ == "__main__":
    main()