from ase.build import sort
from ase.io import read, write
from ase.io.vasp import write_vasp_xdatcar
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--temperature", type=str, required=True)
args = parser.parse_args()

images = read('../bussi_md/{}/bussi.traj'.format(args.temperature), index=':')
write_vasp_xdatcar('../bussi_md/{}/XDATCAR'.format(args.temperature), images)
