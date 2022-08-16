## Open a terminal and do the following steps

### Step 1: SSH into ADA
ssh -X lalitha.v@ada.iiit.ac.in

### Step 2: Change directory
cd General_KO_Codes/

### Step 3: Request a GNode
sinteractive -c 20 -g 4 
(It may take a couple of minutes. Please wait)

Your terminal will now say
```
(base) lalitha.v@gnodexx:~/General_KO_Codes$
```

### To test the Berman/NN codes, run one of the following commands
For testing C(n=2,r=1,m=6)
/home/lalitha.v/miniconda3/envs/venv/bin/python3.8 -m app.auto_tester c_n2_r1_m6

For testing C(n=3,r=1,m=4)
/home/lalitha.v/miniconda3/envs/venv/bin/python3.8 -m app.auto_tester c_n3_r1_m4

For testing C(n=3,r=2,m=4)
/home/lalitha.v/miniconda3/envs/venv/bin/python3.8 -m app.auto_tester c_n3_r2_m4


### To test the NN codes, run one of the following commands.
### Currently there is no need to train any of the NN codes. 


## Dependency
- numpy (1.14.1)
- pytorch (1.4)

## Before Training
```
    python data/generate_data.py
```


## C3(2,4)

- To train :
```
    python -m app.auto_trainer c_n3_r2_m4
```

- To test:
```
    python -m app.auto_tester c_n3_r2_m4
```


We are using Belief Propagation from https://github.com/YairMZ/belief_propagation.