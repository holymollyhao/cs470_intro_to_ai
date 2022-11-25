# TeTra

Test-time adaptation on transformers

## Installation

### Set conda environment
```
$ conda env create --file env.yaml
```

## Result reproduction

### 1. Source generation
Trains 1 epoch on each source dataset to finetune weights of models (only 1 epoch is used due to time constraint)

**Modifiable shell variables**
- `LOG_PREFIX` : Log prefix

- `GPUS` : list of GPUs that your server have (starting from 0)


```
$ ./shell_script/baselines.sh 
```

### 2. Test-time Adaptation
Tests the trained source with different methods (LN_TENT, TeTra).

**Modifiable shell variables**
- `LOG_PREFIX` : Log prefix

- `GPUS` : list of GPUs that your server have (starting from 0)

```
$ ./shell_script/results.sh 
```

### 3. Evaluation
Reproduce the results

**Modifiable shell variables**
- `LOG_PREFIX` : Log prefix

- `LOG_DIRECTORY` : Log directory to walk through to search the trained model

```
$ python eval_script.py \
    --regex {LOG_PREFIX} \
    --directory {LOG_DIRECTORY}
```


## Results