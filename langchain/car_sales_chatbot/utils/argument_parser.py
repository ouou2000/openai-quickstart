import argparse

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Car sales chatbot.')
        self.parser.add_argument('--enable_chat', type=str, help='Whether to enable large model chat')

    def parse_arguments(self):
        args = self.parser.parse_args()
        return args
