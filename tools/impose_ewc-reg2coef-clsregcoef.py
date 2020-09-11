'''
 File Created: Sat Sep 05 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
 This script is to impose ewc reg_sigma_prior
 to combine cls_term and reg_term and get the final ewc_weights.
 python tools/impose_ewc-reg2coef-clsregcoef.py \
    --cls2term-path saved_weights/20200905-ewcweights-tune-regsigmaprior/ewc_cls2term-23200.pkl \
    --reg2term-path saved_weights/20200905-ewcweights-tune-regsigmaprior/ewc_reg2term-23200.pkl \
    --clsregterm-path saved_weights/20200905-ewcweights-tune-regsigmaprior/ewc_clsregterm-23200.pkl \
    --reg2coef 1 \
    --clsregcoef 1 \
    --output-path saved_weights/20200905-ewcweights-tune-regsigmaprior/ewc_weights-23200-post.pkl
'''
import argparse
from det3.ops import read_pkl, write_pkl

def main(cls2term, reg2term, clsregterm, reg2coef, clsregcoef, output_path):
    ewc_weights = {name: param + reg2coef * reg2term[name] + clsregcoef * clsregterm[name]
        for name, param in cls2term.items()}
    write_pkl(ewc_weights, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls2term-path", type=str, help="input path of cls2term.pkl")
    parser.add_argument("--reg2term-path", type=str, help="input path of reg2term.pkl")
    parser.add_argument("--clsregterm-path", type=str, help="input path of clsregterm.pkl")
    parser.add_argument("--output-path", type=str, help="output path of ewc_weights.pkl")
    parser.add_argument("--reg2coef", type=float, help="hypeparameter of regression square term")
    parser.add_argument("--clsregcoef", type=float, help="hypeparameter of classification regression production term")
    args = parser.parse_args()
    cls2term = read_pkl(args.cls2term_path)
    reg2term = read_pkl(args.reg2term_path)
    clsregterm = read_pkl(args.clsregterm_path)
    output_path = args.output_path
    reg2coef = args.reg2coef
    clsregcoef = args.clsregcoef
    main(cls2term, reg2term, clsregterm,reg2coef, clsregcoef, output_path)
