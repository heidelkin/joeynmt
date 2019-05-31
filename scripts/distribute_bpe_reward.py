# coding: utf-8

"""
To distribute token-level rewards to sub-word units.
"""

import argparse
import re
import sys

import numpy as np


def main(args):

    target_file = args.target
    reward_file = args.reward

    with open(target_file, 'r') as bpe_in, \
         open(reward_file, 'r') as fbkin, open('feedback.bpe', 'w') as fbkout:

        trans = bpe_in.readlines()
        rewards = fbkin.readlines()

        for idx, (sent, reward) in enumerate(zip(trans, rewards)):
            reward_tok = reward.split()

            if '@@' not in sent:
                # No sub-words in this sentence
                bpe_reward = reward_tok
            else:
                sent_str = sent.replace("@@ ", "")
                sent_tok = sent.split()
                assert(len(sent_str.split())==len(reward_tok)), \
                    "input sentence length should be equal to number of reward"

                if idx % args.print_interval == 0:
                    print("sentence (post-process bpe):{}".format(
                          sent_str[:-1]))
                    print('input reward:{}'.format(" ".join(reward_tok)))

                loc_bpe = [loc for loc, tok in enumerate(sent_tok) 
                            if "@@" in tok]
                loc_bpe_in_original = [bpeloc - len(loc_bpe[:idx]) 
                                        for idx, bpeloc in enumerate(loc_bpe)]
                loc_bpe_suffix = [loc + 1 for loc in loc_bpe]

                bpe_reward = [None] * len(sent_tok)
                bpe_count = 0
                for ii in range(len(sent_tok)):
                    if ii in loc_bpe_suffix:
                        # the bpe suffix has the same reward as the bpe-prefix
                        orig_loc_bpe_prefix = loc_bpe_in_original[loc_bpe_suffix.index(ii)]
                        bpe_reward[ii] = reward_tok[orig_loc_bpe_prefix]
                        if idx % args.print_interval == 0:
                            print('BPE suffix! loc:{} ; token[loc]:{}; orig_loc:{}; '
                                  'reward[orig_loc]:{}'.format(ii, sent_tok[ii], 
                                  orig_loc_bpe_prefix, reward_tok[orig_loc_bpe_prefix]))
                        bpe_count += 1
                    else:
                        bpe_reward[ii] = reward_tok[ii-bpe_count]

                if idx % args.print_interval == 0:
                    print('sentence (with sub-word):{}'.format(" ".join(sent_tok)))
                    print("bpe reward:{}\n".format(" ".join(bpe_reward)))

            fbkout.write(" ".join(bpe_reward) + "\n")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument("target", type=str, 
                    help="File for MT output after BPE preprocessing")
    ap.add_argument("reward", type=str, 
                    help="File for storing rewards of the MT output" 
                         "before BPE preprocessing")
    ap.add_argument("--output", type=str, default="bpe_reward.feedback",
                    help="File name for BPE-handled rewards")
    ap.add_argument("--print_interval", type=int, default=1000,
                    help="Frequency for printing the reward "
                         "distribution processes")

    args = ap.parse_args()

    main(args)
