import configparser
import argparse
import logging
import os
import warnings
import torch
from mpi4py import MPI
from fl import FL


# For MPI experiments
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()


def main():
    is_mpi = COMM.Get_size() != 1
    config = read_config()
    fl = FL(config, is_mpi, RANK)
    fl.start()



def read_config():
  config = configparser.ConfigParser()
  config.read('config/opp/dccae/A0_B10_AB30_label_B_test_A')
  return config
config = read_config()
fl = FL(config)
fl.start()


if __name__ == "__main__":
    main()
