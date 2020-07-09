'''
 File Created: Thu Jul 09 2020
 Author: Peng YUN (pyun@ust.hk)
 Copyright 2018-2020 Peng YUN, RAM-Lab, HKUST
'''
import argparse
from incdet3.main import load_config_file
from det3.ops import read_txt, write_txt
from incdet3.utils import bcolors

class StoreDictKeyPair(argparse.Action):
     def __call__(self, parser, namespace, values, option_string=None):
         key_pairs = {}
         for kv in values.split(","):
             k,v = kv.split("=")
             key_pairs[k] = v
         setattr(namespace, self.dest, key_pairs)

if __name__ == "__main__":
    # args:
    parser = argparse.ArgumentParser(
        description='create a cfg file according to a template file and change key pairs.')
    parser.add_argument("--template-cfg-path", type=str)
    parser.add_argument("--key-pairs", dest="key_pairs", action=StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...")
    # (org_line)=(line)
    parser.add_argument("--save-path", type=str)
    args = parser.parse_args()
    template_cfg_path = args.template_cfg_path
    key_pairs = args.key_pairs if args.key_pairs is not None else {}
    save_path = args.save_path
    # read_cfg with read_txt
    cfg_txt = read_txt(template_cfg_path)
    # change according key_paris
    num_of_changes = 0
    for org_line, change_line in key_pairs.items():
        num_of_change_ = 0
        for i, line in enumerate(cfg_txt):
            start = line.find(org_line)
            if start == -1:
                continue
            else:
                num_of_change_ += 1
                tmp = ""+line
                cfg_txt[i] = line[:start] + change_line + line[start+len(org_line):]
                print(f"line [{i+1}] \n" +\
                      f"{tmp} \n"+\
                      bcolors.BOLD+ f"{cfg_txt[i]}" + bcolors.ENDC)
        if num_of_change_ != 1:
            print(bcolors.WARNING)
            assert num_of_change_ != 0
        print(f"This line have {num_of_change_} changes"+bcolors.ENDC)
        num_of_changes += num_of_change_
    # write_cfg with write_txt
    print(bcolors.BOLD + save_path + f" totally have {num_of_changes} changes" + bcolors.ENDC)
    write_txt(cfg_txt, save_path)
    