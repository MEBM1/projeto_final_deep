import configparser
import argparse
import sys 

#sys.path.insert(0,r'C:\Users\madua\Documents\Mestrado\Deep Learning\Projeto Final\looking-main\utils')
#import sys
#sys.path.insert(0,r'C:\Users\madua\Documents\Mestrado\Deep Learning\Projeto Final\looking-main\utils')
sys.path.insert(0,r'C:\Users\madua\Documents\Mestrado\Deep Learning\Projeto Final\looking-main\utils')
from trainer import *
#from utils.trainer import *

parser_command = argparse.ArgumentParser(description='Training the model')
parser_command.add_argument('--file', dest='f', type=str, help='Config file name to use', default="config.ini")

args = parser_command.parse_args()
parser_file = args.f

config = configparser.ConfigParser()
config.read(parser_file)

parser = Parser(config)
parser.parse()

trainer = Trainer(parser)
trainer.train()