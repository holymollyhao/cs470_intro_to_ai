# TeTra

Test-time adaptation on transformers

## Prerequisites

### Change current directory to project folder
```
$ cd ./cs470_intro_to_ai
```
### Set conda environment
```
$ conda env create --file env.yaml
```
### Create directories
```
$ mkdir log/
$ mkdir raw_logs/
$ mkdir cache/
$ mkdir dataset/
```
### Unzip the Dataset folder
```
Download the finefood, sst-2, imdb dataset from the link below :
https://drive.google.com/file/d/1Ll0B1EL53AitPQBe8pPUHNopp-kIw8o4/view?usp=sharing

Unzip them inside the ./dataset folder
```


## Result reproduction

### 1. Edit the shell file
Edit the function below accordingly, for a single gpu instance, set local_default_num_jobs (number of GPUs on server = 1),
and local_num_max_jobs (total number of jobs to run concurrently = 1).
```
wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local default_num_jobs=8 #12
  local num_max_jobs=8
  echo $num_max_jobs
  if ((${#background[@]} >= num_max_jobs)); then
    wait -n
  fi
}
```

### 2. Source generation
Trains 1 epoch on each source dataset to finetune weights of models (only 1 epoch is used due to time constraint)

**Modifiable shell variables**
- `LOG_PREFIX` : Log prefix

- `GPUS` : list of GPUs that your server have (starting from 0)


```
$ ./shell_script/baselines.sh 
```

### 3. Test-time Adaptation
Tests the trained source with different methods (LN_TENT, TeTra).

**Modifiable shell variables**
- `LOG_PREFIX` : Log prefix

- `GPUS` : list of GPUs that your server have (starting from 0)

- `MODELS` : list of models that you will test upon (i.e. distilbert, bert)
- `SEED` : list of seeds that you will test upon

Currently, the `results.sh` is implemented s.t. it only tests out distilbert on seed 0.
Note that all of the results where derived from 8 * NVIDIA GeForce RTX 3090 GPUs.
```
$ ./shell_script/results.sh 
```

### 3. Evaluation
Reproduce the results

**Modifiable shell variables**
- `LOG_PREFIX` : Log prefix

- `LOG_DIRECTORY` : Log directory to walk through to search the trained model

Currently, the `eval_script.py` is implemented s.t. it only shows results of distilbert on seed 0.
```
$ python eval_script.py \
    --regex {LOG_PREFIX} \
    --directory {LOG_DIRECTORY}
```
For reproduction purposes, we have set the default arguments accordingly with the baseline and test shell files, thus you can get the results simply by running:
```
$ python eval_script.py
```

