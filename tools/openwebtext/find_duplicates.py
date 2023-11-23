# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import argparse
from functools import partial
import itertools
import json
from lsh import cache, minhash
import multiprocessing
import numpy as np
import time
import pickle
import sys
import os
from pydebug import gd, infoTensor
gd.debuginfo(prj="mt")
# This function is adapted from:
#   https://github.com/mattilyra/LSH/blob/master/examples/Introduction.ipynb
def shingles(text, char_ngram=5):
    return set(text[head:head + char_ngram]
               for head in range(0, len(text) - char_ngram))


# This function is adapted from:
#  https://github.com/mattilyra/LSH/blob/master/examples/Introduction.ipynb
def jaccard(set_a, set_b, args):
    if len(set_a) < 1 or len(set_b) < 1:
        return 0.0

    intersection = set_a & set_b
    union = set_a | set_b

    if args.jaccard == 'min':
        return len(intersection) / min(len(set_a), len(set_b))
    elif args.jaccard == 'max':
        return len(intersection) / max(len(set_a), len(set_b))
    else:
        return len(intersection) / len(union)

def compute_fingerprint(line, key):
    try:
        myjson = json.loads(line)
        url = myjson[key]
        text = myjson['text']
        fingerprint = hasher.fingerprint(text)
    except Exception as e:
        gd.debuginfo(prj="mt", info=f'Error: {e}')
        return None, None, None, False

    return url, text, fingerprint, True

def url_pairs_to_remove(args, bucket_urls, url_doc):
    remove_urls_list = []
    deduped_local, counter_local = 0, 0
    iteration = 0
    while len(bucket_urls) > 1:
        if args.heuristic_iter != -1 and \
            iteration == args.heuristic_iter:
            break

        items = list(bucket_urls)
        remove_urls = []
        main_url = items[np.random.randint(0, len(items))]
        main_dhingles = shingles(url_doc[main_url])

        for i in range(0, len(items)):
            counter_local += 1
            other_url = items[i]
            if other_url == main_url:
                continue
            other_shingles = shingles(url_doc[other_url])
            try:
                jaccard_sim = jaccard(main_dhingles, other_shingles, args)
            except Exception as e:
                gd.debuginfo(prj="mt", info=f'Error:{e}')
                jaccard_sim = 0.0
            if jaccard_sim > 0.5:
                remove_urls.append({other_url: jaccard_sim})
                deduped_local += 1
                bucket_urls.remove(other_url)

        bucket_urls.remove(main_url)
        if len(remove_urls) > 0:
            remove_urls_list.append({main_url: remove_urls})
        iteration += 1
    return remove_urls_list, deduped_local, counter_local

def write_remove_urls_list(remove_urls_list, f_out):
    if len(remove_urls_list) > 0:
        for each_url_remove in remove_urls_list:
            myjson = json.dumps(each_url_remove, ensure_ascii=False)
            f_out.write(myjson.encode('utf-8'))
            f_out.write('\n'.encode('utf-8'))

def compute_jaccard(each_bin, num_bins, start_time_local):

    remove_urls_list = []
    deduped_local, counter_local, bucket_local = 0, 0, 0

    for bucket_id in each_bin:
        bucket_local += 1
        if os.getpid() % num_bins == 0 and bucket_local % 100000 == 0:
            gd.debuginfo(prj="mt", info=f"Counter {bucket_local}, "
                                        f"progress {float(bucket_local)/float(len(each_bin)):.2f} "
                                        f"time {time.time() - start_time_local:.2f}")

        if len(each_bin[bucket_id]) <= 1:
            continue

        bucket_urls = each_bin[bucket_id].copy()
        remove_urls_list_sub, deduped_local_sub, counter_local_sub = \
            url_pairs_to_remove(args, bucket_urls, url_doc)

        deduped_local += deduped_local_sub
        counter_local += counter_local_sub
        if len(remove_urls_list_sub) > 0:
            remove_urls_list.extend(remove_urls_list_sub)

    return remove_urls_list, deduped_local, counter_local

def find_pair_urls_parallel(args, lshcache, url_doc):
    start_time = time.time()
    f_out = open(args.output, 'wb')
    deduped, counter = 0, 0

    # compute jaccards of buckets in bin in parallel (parallelism
    # limited to # of bins)
    num_bins = len(lshcache.bins)
    pool = multiprocessing.Pool(num_bins)
    compute_jaccard_partial = partial(compute_jaccard, num_bins=num_bins, \
        start_time_local=start_time)
    # don't need to pass args and url_doc as they are already shared
    compute_jaccard_iter = pool.imap(compute_jaccard_partial, lshcache.bins)

    gd.debuginfo(prj="mt", info=f"multiprocessing init took {time.time() - start_time:.2f}")
    for remove_urls_list, deduped_local, counter_local in compute_jaccard_iter:
        deduped += deduped_local
        counter += counter_local
        write_remove_urls_list(remove_urls_list, f_out)
        gd.debuginfo(prj="mt", info=f' [write]> processed {counter} documents '
                                    f' in {time.time() - start_time:.2f} '
                                    f' eoncds and deduped {deduped} documents ...')

    pool.close()
    pool.join()
    f_out.close()

    gd.debuginfo(prj="mt", info=f' Taken time for jaccard similariries {time.time() - start_time:.2f} seconds')

def find_pair_urls_sequential(args, lshcache, url_doc):
    start_time = time.time()
    f_out = open(args.output, 'wb')
    deduped, counter = 0, 0
    for b in lshcache.bins:
        for bucket_id in b:
            if len(b[bucket_id]) <= 1:
                continue

            bucket_urls = b[bucket_id].copy()
            remove_urls_list_sub, deduped_local_sub, counter_local_sub = \
                url_pairs_to_remove(args, bucket_urls, url_doc)

            deduped += deduped_local_sub
            counter += counter_local_sub
            write_remove_urls_list(remove_urls_list_sub, f_out)
            if counter % 10000 == 0:
                gd.debuginfo(prj="mt",
                             info=f' [write]> processed {counter} documents '
                                  f' in {time.time() - start_time:.2f} seoncds '
                                  f'and deduped {deduped} documents ...')
    f_out.close()
    gd.debuginfo(prj="mt", info=f' [write]> processed {counter} documents '
                                f'in {time.time() - start_time:.2f} '
                                f'seoncds and deduped {deduped} documents ...')

if __name__ == '__main__':

    gd.debuginfo(prj="mt", info=f'parsing the arguments ...')

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234,
                       help='Random seed used for python, numpy')
    parser.add_argument('--inputs', nargs = '*', default=None, help = \
                        'Pairwise list of the input files and keys, '
                        'e.g. --inputs cc.json cc_id news.json news_id')
    parser.add_argument('--load-fingerprints', nargs = '*', default=None,
                       help='Load fingerprints from a list of pickle files,'
                        ' e.g. cc.pkl news.pkl')
    parser.add_argument('--save-fingerprints', type=str, default=None,
                       help='Save the fingerprints of the inputs.')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file name that consists of all ids'
                        ' with matching similarities')
    parser.add_argument('--jaccard', type=str, default='union',
                        choices=['union', 'min', 'max'], help='Jaccard'\
                        ' similarity computation')
    parser.add_argument('--heuristic-iter', type=int, default=1,
                       help='Number of iterations to run the heuristics'
                        ': use -1 for exact')
    parser.add_argument('--num-bands', type=int, default=10,
                       help='Number of bands to use in cache')
    parser.add_argument('--num-seeds', type=int, default=100,
                       help='Number of seeds to use for minhash. Note that'
                        ' this value should be divisible by num-bands')
    parser.add_argument('--jaccard-parallel', action='store_true',
                       help='Use this to process large number of documents.')
    args = parser.parse_args()

    gd.debuginfo(prj="mt", info=f'finding possible duplicate content ...')

    # set seed and get an array of seeds of 100 integers
    np.random.seed(args.seed)
    seeds = np.random.randint(0, 1e6, size=args.num_seeds)

    # initialize minhash and lsh cache
    hasher = minhash.MinHasher(seeds=seeds, char_ngram=5, hashbytes=4)
    lshcache = cache.Cache(num_bands=args.num_bands, hasher=hasher)

    url_doc = {}

    # load fingerprints from pickle file if needed
    if args.load_fingerprints is not None:
        for count_fp, fp_file_name in enumerate(args.load_fingerprints):
            gd.debuginfo(prj="mt", info=f"Loading fingerprints from pickle file {fp_file_name}")
            fp = open(fp_file_name, "rb")
            if count_fp == 0:
                # assign directory for the first pkl
                lshcache = pickle.load(fp)
                url_doc = pickle.load(fp)
            else:
                # append these to lshcache and url_doc
                local_lshcache = pickle.load(fp)
                local_url_doc = pickle.load(fp)
                for url in local_lshcache.fingerprints.keys():
                    url_doc[url] = local_url_doc[url]
                    lshcache.add_fingergd.debuginfo(prj="mt", info=f'{local_lshcache.fingerprints[url]}')
            fp.close()

    counter = 0
    start_time = time.time()

    # compute finger prints of the inputs if any
    # input file and the key to use as id
    if args.inputs is not None:
        gd.debuginfo(prj="mt", info=f"Computing fingerprints")
        assert len(args.inputs) % 2 == 0
        for input_file, key in zip(args.inputs[::2], args.inputs[1::2]):
            gd.debuginfo(prj="mt", info=f' document processing {input_file} with key {key}')

            # compute fingerprints in parallel
            num_workers = 40
            pool = multiprocessing.Pool(num_workers)
            fin = open(input_file, 'r', encoding='utf-8')
            compute_fingerprint_partial = partial(compute_fingerprint, key=key)
            compute_fingerprint_iter = pool.imap(compute_fingerprint_partial,
                                                    fin, 512)
            # traverse all the texts and add fingerprints
            for url, text, fingerprint, flag in compute_fingerprint_iter:
                gd.debuginfo(prj="mt", info=f'fingerprint={fingerprint}, url={url}, text={text}')
                counter += 1
                if flag:
                    url_doc[url] = text
                    lshcache.add_fingerprint()

                if counter % 10000 == 0:
                    gd.debuginfo(prj="mt", info=f' [read]> processed {counter} documents '
                                                f' in {time.time() - start_time:.2f} seconds ...')

            fin.close()
            pool.close()
            pool.join()

    # Save the fingerprints if needed
    if args.save_fingerprints is not None:
        gd.debuginfo(prj="mt", info=f"Saving fingerprints to pickle file {args.save_fingerprints}")
        with open(args.save_fingerprints, 'wb') as f_save:
            pickle.dump(lshcache, f_save)
            pickle.dump(url_doc, f_save)

    # compute jaccard index of the input texts and write to file if needed
    if args.output is not None:
        gd.debuginfo(prj="mt", info=f"Compute jaccard similarity")
        if args.jaccard_parallel:
            find_pair_urls_parallel(args, lshcache, url_doc)
        else:
            find_pair_urls_sequential(args, lshcache, url_doc)

    gd.debuginfo(prj="mt", info=f'done :-)')
 
