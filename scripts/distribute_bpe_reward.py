# coding: utf-8

"""
To distribute token-level rewards to sub-word units.

Example: 
    Sentence with tokenized words but without bpe: ["Today", "is", "not"]
    Corresponding rewards: ["1", "0", "1"]
    
    Preprocess the above file using trained bpe:
    bpe = ["To@@ ", "day", "is", "n@@", t"]

    Output (Reward): ["1", "1", "0", "1", "1"]

Define: bpe_suffix - the token split after "@@"
        bpe_prefix - the part with "@@" 
        Example:
            "To@@" is bpe_prefix where "day" is its suffix

        unsplit - Token in non-bpe format, e.g. "To@@ " and "day"
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
        assert(len(trans)==len(rewards))

        for idx, (sent, reward) in enumerate(zip(trans, rewards)):
            reward_tok = reward.split()

            if '@@' not in sent:
                # No sub-words in this sentence
                bpe_reward = reward_tok
            else:
                sent_unsplit_str = sent.replace("@@ ", "")
                sent_tok = sent.split()
                assert(len(sent_unsplit_str.split())==len(reward_tok)), \
                    "Input sentence without BPE splitting should be of "\
                    "equal length to number of reward"

                if idx % args.print_interval == 0:
                    print("sentence (post-process bpe):{}".format(
                          sent_unsplit_str[:-1]))
                    print('input reward:{}'.format(" ".join(reward_tok)))

                location_bpe_prefix = [loc for loc, tok in enumerate(sent_tok) 
                                        if "@@" in tok]
                
                location_bpe_unsplit = [] # their locations without bpe split
                for idx, loc_bpe_prefix in enumerate(location_bpe_prefix):
                    number_previous_bpe = len(location_bpe_prefix[:idx])
                    loc_bpe_unsplit = loc_bpe_prefix - number_previous_bpe
                    location_bpe_unsplit.append(loc_bpe_unsplit)

                location_bpe_suffix = [loc + 1 for loc in location_bpe_prefix]

                bpe_reward = [None] * len(sent_tok)
                bpe_count = 0
                for ii in range(len(sent_tok)):
                    # bpe suffix has the same reward as its prefix
                    if ii in location_bpe_suffix:
                        ## Get reward by its locaation, same as its prefix, 
                        ## in the unsplit format.
                        which_bpe_suffix = location_bpe_suffix.index(ii)
                        loc_bpe_unsplit = location_bpe_unsplit[which_bpe_suffix]
                        bpe_reward[ii] = reward_tok[loc_bpe_unsplit]
                        if idx % args.print_interval == 0:
                            print('BPE suffix! loc:{} ; token[loc]:{}; orig_loc:{}; '
                                  'reward[orig_loc]:{}'.format(ii, sent_tok[ii], 
                                  loc_bpe_unsplit, reward_tok[loc_bpe_unsplit]))
                        bpe_count += 1
                    else:
                        bpe_reward[ii] = reward_tok[ii-bpe_count]

                assert(bpe_count == len(location_bpe_prefix) == len(location_bpe_suffix))

                if idx % args.print_interval == 0:
                    print('sentence (with sub-word):{}'.format(" ".join(sent_tok)))
                    print("bpe reward:{}\n".format(" ".join(bpe_reward)))

            fbkout.write(" ".join(bpe_reward) + "\n")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument("target", type=str, 
                    help="File for MT output after preprocessing"
                         "tokenized with BPE")
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
