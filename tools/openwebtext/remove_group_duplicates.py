# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.


import json
import time
import sys
from pydebug import gd, infoTensor
gd.debuginfo(prj="mt")

if __name__ == '__main__':

    url_filename = sys.argv[1]
    data_filename = sys.argv[2]
    output_filename = sys.argv[3]

    urls = set()
    with open(url_filename, 'r') as f:
        for line in f:
            myjson = json.loads(line)
            for key in myjson:
                this_urls = myjson[key]
                for i in range(1, len(this_urls)):
                    urls.add(this_urls[i])
    gd.debuginfo(prj="mt", info=f'will be removing {len(urls)} urls')

    written_docs = 0
    removed_docs = 0
    removed_chars = 0
    start_time = time.time()
    with open(output_filename, 'wb') as fout:
        with open(data_filename, 'r') as fin:
            for line in fin:
                try:
                    myjson = json.loads(line)
                    url = myjson['url']
                    if url in urls:
                        gd.debuginfo(prj="mt", info=f'removing', myjson)
                        removed_docs += 1
                        removed_chars += len(myjson['text'])
                        continue
                    myjson = json.dumps(myjson, ensure_ascii=False)
                    fout.write(myjson.encode('utf-8'))
                    fout.write('\n'.encode('utf-8'))
                    written_docs += 1
                    if written_docs % 10000 == 0:
                        gd.debuginfo(prj="mt", info=f' [PROCESSED] time (s): {:.2f} | written: {} '
                              '| removed: {} (char: {})'.format(
                                  time.time() - start_time,
                                  written_docs, removed_docs, removed_chars))
                except Exception as e:
                    gd.debuginfo(prj="mt", info=f'[SKIPPING]', line, e)

    gd.debuginfo(prj="mt", info=f' [PROCESSED] time (s): {:.2f} | written: {} '
          '| removed: {} (char: {})'.format(
              time.time() - start_time,
              written_docs, removed_docs, removed_chars))
    gd.debuginfo(prj="mt", info=f'done :-)')
