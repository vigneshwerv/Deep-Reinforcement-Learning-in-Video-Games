import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--train', action='store_true', default=False,
            help='Set the agent in training mode.')
    g.add_argument('--test', action='store_true', default=False,
                help='Set the agent in testing mode.')
    parser.add_argument('--name', action='store', default=None, type=str,
            required=True, help="""Set the name of the agent.
                                   
                                   If the name of the agent exists in the logs
                                   directory, then the saved agent file is used.
                                   Otherwise, a new agent save file is created.""")
    args = parser.parse_args()
    from DQNController import DQNController

    controller = DQNController(args=args)
    if args.train:
        controller.run()
    elif args.test:
        controller.run_testing_stage()
