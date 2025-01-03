import configparser
import argparse
from utils.trainer import *
import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

parser_command = argparse.ArgumentParser(description='Training the model')
parser_command.add_argument('--file', dest='f', type=str, help='Config file name to use', default="config.ini")

args = parser_command.parse_args()
parser_file = args.f

config = configparser.ConfigParser()
config.read(parser_file)

parser = Parser(config)
parser.parse()

evaluator = Evaluator(parser)
#evaluator.evaluate()
evaluator.evaluate_with_ann()