# How to use the DTU HPC


### How to get there

- SSH Login with felteu  on login.gbar.dtu.dk
- change to HPC node with `qrsh`
- node for data transfer: `scp file userid@transfer.gbar.dtu.dk:directory/`


### Jobs
- use `bsub` to submit jobs
- scratch is `/work3/felteu`
- `bstat -M` to see currently running jobs
- `bpeek JOBID`  to see stdout

### Python
- `module load python3`
- `virtualenv env` works
- for python3 run pip using `python3 -m pip install`

### Status
- `nodestat -f gpuv100` to see node usage
- `bjobs -u all -q gpuv100` for queue status