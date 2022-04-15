#服务端代码
import sys
import asyncio
import itertools
import functools
from sanic import Sanic
from sanic.response import json, text
from sanic.log import logger
from sanic.exceptions import ServerError

import sanic
import threading
import io
import torch
import logging
import numpy as np
# from cyclegan import get_pretrained_model

app = Sanic(__name__)

device = torch.device('cpu')
# we only run 1 inference run at any time (one could schedule between several runners if desired)
MAX_QUEUE_SIZE = 200  # we accept a backlog of MAX_QUEUE_SIZE before handing out "Too busy" errors
MAX_BATCH_SIZE = 100  # we put at most MAX_BATCH_SIZE things in a single batch
MAX_WAIT = 10       # we wait at most MAX_WAIT seconds before running for more inputs to arrive in batching

class HandlingError(Exception):
    def __init__(self, msg, code=500):
        super().__init__()
        self.handling_code = code
        self.handling_msg = msg

class ModelRunner:
    def __init__(self):
        self.queue = []
        self.queue_lock = None
        self.model = torch.jit.load("/root/Transformer_model/TinyBERT_4L_zh.pt")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.needs_processing = None

        self.needs_processing_timer = None

    def schedule_processing_if_needed(self):
        if len(self.queue) >= MAX_BATCH_SIZE:
            logger.debug("next batch ready when processing a batch")
            self.needs_processing.set()
        elif self.queue:
            logger.debug("queue nonempty when processing a batch, setting next timer")
            self.needs_processing_timer = app.loop.call_at(self.queue[0]["time"] + MAX_WAIT, self.needs_processing.set)

    async def process_input(self, input):
        our_task = {"done_event": asyncio.Event(loop=app.loop),
                    "input": input,
                    "time": app.loop.time()}
        async with self.queue_lock:
            if len(self.queue) >= MAX_QUEUE_SIZE:
                raise HandlingError("I'm too busy", code=503)
            self.queue.append(our_task)
            logger.debug("enqueued task. new queue size:{}".format(len(self.queue)))
            self.schedule_processing_if_needed()

        await our_task["done_event"].wait()
        return our_task["output"]

    def run_model(self, batch):  # runs in other thread
        print("## 消费 ##:{}".format(batch))
        query_embeddings = self.model(batch.to(device))
        res_embeddings = np.asarray(query_embeddings[1].detach().cpu())
        return res_embeddings.tolist()

    async def model_runner(self):
        self.queue_lock = asyncio.Lock(loop=app.loop)
        self.needs_processing = asyncio.Event(loop=app.loop)
        # logger.info("started model runner for {}".format(self.model_name))
        while True:
            await self.needs_processing.wait()
            self.needs_processing.clear()
            if self.needs_processing_timer is not None:
                self.needs_processing_timer.cancel()
                self.needs_processing_timer = None
            async with self.queue_lock:
                if self.queue:
                    longest_wait = app.loop.time() - self.queue[0]["time"]
                else:  # oops
                    longest_wait = None
                logger.debug("launching processing. queue size:{}.longest wait:{}".format(len(self.queue), longest_wait))
                to_process = self.queue[:MAX_BATCH_SIZE]
                del self.queue[:len(to_process)]
                self.schedule_processing_if_needed()
            # so here we copy, it would be neater to avoid this
            # batch = torch.stack([t["input"] for t in to_process], dim=0)
            print("&&{}&&".format(to_process))
            # batch = torch.stack([t in to_process], dim=0)
            batch = torch.stack([t["input"] for t in to_process], dim=0)
            print("**{}**".format(batch))
            # we could delete inputs here ...

            result = await app.loop.run_in_executor(None, functools.partial(self.run_model, batch))
            for t, r in zip(to_process, result):
                t["output"] = r
                t["done_event"].set()
            del to_process

style_transfer_runner = ModelRunner()#sys.argv[1]

@app.route('/test', methods=['POST'])
async def image(request):
    try:
        json_str = request.json
        print("Req:{}".format(json_str["instances"][0]["ori_input_quests"]))
        res = await style_transfer_runner.process_input(torch.LongTensor(json_str["instances"][0]["ori_input_quests"]))
        return json({'res':res})
        # return sanic.response.raw("{status:'200'}", status=200,content_type='json')
        #out_im = await style_transfer_runner.process_input(im)
        #return sanic.response.raw("", status=200,content_type='json')
    except HandlingError as e:
        # we don't want these to be logged...
        return sanic.response.text(e.handling_msg, status=e.handling_code)

app.add_task(style_transfer_runner.model_runner())
app.run(host="0.0.0.0", port=8000,debug=True)

# curl -X POST http://127.0.0.1:8000/test -d '{"query": "你不要诈骗了","org": [{"org_name": "开场白","cls_score": 0.7,"rule_score": 0.7,"intent_list": []}]}'
# curl -X POST http://127.0.0.1:8000/test -d '{"instances": [{"ori_input_quests": [101,1998,3739,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}]}'

# curl -X POST http://127.0.0.1:8000/test -d '{"instances": [{"ori_input_quests": [101,1998,3739,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}]}' && curl -X POST http://127.0.0.1:8000/test -d '{"instances": [{"ori_input_quests": [101,1998,3739,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}]}' && curl -X POST http://127.0.0.1:8000/test -d '{"instances": [{"ori_input_quests": [101,1998,3739,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}]}'