# SoLM

Source code for SoLM in paper: 

**Enhancing Rumor Detection Methods with Propagation Structure Infused Language Model**

## Run

The pretraining code can be run in the following ways:

```shell script
nohup python main\(mlm\).py --gpu 4 &
nohup python main\(mlm\)\(parallel\).py &
nohup python -m torch.distributed.launch --nproc_per_node=7 --use_env main\(mlm\)\(ddp\).py &
nohup python main\(prp\).py --gpu 0 &
```

## Dependencies

- [pytorch](https://pytorch.org/) == 1.12.0

- [transformers](https://github.com/huggingface/transformers) == 4.2.1

- [datasets](https://github.com/huggingface/datasets) == 2.10.1