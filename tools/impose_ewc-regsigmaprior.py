'''
 File Created: Sat Sep 05 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
 This script is to impose ewc reg_sigma_prior
 to combine cls_term and reg_term and get the final ewc_weights.
 python tools/impose_ewc-regsigmaprior.py \
    --cls-term-path saved_weights/20200905-ewcweights-tune-regsigmaprior/ewc_clsterm-23200.pkl\
    --reg-term-path saved_weights/20200905-ewcweights-tune-regsigmaprior/ewc_regterm-23200.pkl\
    --reg-sigma-prior 1\
    --output-path saved_weights/20200905-ewcweights-tune-regsigmaprior/ewc_weights-23200-post.pkl
'''
import argparse
from det3.ops import read_pkl, write_pkl

def main(cls_term, reg_term, reg_sigma_prior, output_path):
    ewc_weights = {name: param + reg_term[name]/reg_sigma_prior
        for name, param in cls_term.items()}
    write_pkl(ewc_weights, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls-term-path", type=str, help="input path of cls_term.pkl")
    parser.add_argument("--reg-term-path", type=str, help="input path of reg_term.pkl")
    parser.add_argument("--output-path", type=str, help="output path of ewc_weights.pkl")
    parser.add_argument("--reg-sigma-prior", type=float, help="hypeparameter of regression term sigma prior")
    args = parser.parse_args()
    cls_term = read_pkl(args.cls_term_path)
    reg_term = read_pkl(args.reg_term_path)
    output_path = args.output_path
    reg_sigma_prior = args.reg_sigma_prior
    main(cls_term, reg_term, reg_sigma_prior, output_path)
