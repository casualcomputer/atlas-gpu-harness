# Attribution

- **PyTorch** (`torch.sort`, `torch.argsort`, `torch.topk`, `torch.cuda.Event`): used for GPU operator implementations and timing. Copyright and license belong to the PyTorch contributors. [Repository and license](https://github.com/pytorch/pytorch/blob/main/LICENSE), [sort documentation](https://docs.pytorch.org/docs/stable/generated/torch.sort.html), [CUDA event documentation](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html).
- **PyMongo**: official MongoDB Python driver. [Documentation](https://www.mongodb.com/docs/languages/python/pymongo-driver/current/), [license](https://github.com/mongodb/mongo-python-driver/blob/master/LICENSE).
- **filelock**: local process lock for one GPU worker and one controller per run. [Repository](https://github.com/tox-dev/filelock).
- **python-dotenv**: reads an ignored local `.env` file. [Repository](https://github.com/theskumar/python-dotenv).
- **mongomock**: used only by controller tests; it does not establish Atlas integration. [Repository](https://github.com/mongomock/mongomock).

The Strata `qsa_select` benchmark was considered in the project plan. Its CUDA implementation is large and would require build/toolchain work; no Strata code or results were copied into this repository. The verified desktop had a working CUDA PyTorch environment, so this MVP uses that path.
