'''
 File Created: Sat Sep 05 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
 This script is to impose ewc reg_coef
 to combine cls_term and reg_term and get the final mas_weights.
 python tools/impose_mas-regcoef.py \
    --cls-term-path saved_weights/20201005-masweights-biased128/mas_newclsterm-23200.pkl \
    --reg-term-path saved_weights/20201005-masweights-biased128/mas_newregterm-23200.pkl \
    --reg-coef 3 \
    --output-path saved_weights/20201006-masweights-tuneregcoef/mas_omega_regcoef3-23200.pkl
'''
import argparse
from det3.ops import read_pkl, write_pkl

def main(cls_term, reg_term, reg_coef, output_path):
    mas_weights = {name: param + reg_term[name] * reg_coef
        for name, param in cls_term.items()}
    write_pkl(mas_weights, output_path)
    print(f"{output_path} has been saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls-term-path", type=str, help="input path of cls_term.pkl")
    parser.add_argument("--reg-term-path", type=str, help="input path of reg_term.pkl")
    parser.add_argument("--output-path", type=str, help="output path of mas_weights.pkl")
    parser.add_argument("--reg-coef", type=float, help="hypeparameter of regression term")
    args = parser.parse_args()
    cls_term = read_pkl(args.cls_term_path)
    reg_term = read_pkl(args.reg_term_path)
    output_path = args.output_path
    reg_coef = args.reg_coef
    main(cls_term, reg_term, reg_coef, output_path)
