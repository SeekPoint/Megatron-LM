# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import sys
import json
import requests

from pydebug import gd, infoTensor
gd.debuginfo(prj="mt")

if __name__ == "__main__":
    url = sys.argv[1]
    url = 'http://' + url + '/api'
    headers = {'Content-Type': 'application/json'}

    while True:
        sentence = input("Enter prompt: ")
        tokens_to_generate = int(eval(input("Enter number of tokens to generate: ")))

        data = {"prompts": [sentence], "tokens_to_generate": tokens_to_generate}
        response = requests.put(url, data=json.dumps(data), headers=headers)

        if response.status_code != 200:
            gd.debuginfo(prj="mt", info=f"Error {response.status_code}: {response.json()['message']}")
        else:
            gd.debuginfo(prj="mt", info=f"Megatron Response: ")
            gd.debuginfo(prj="mt", info=fresponse.json()['text'][0])
