[![Build Status](https://travis-ci.org/cui-unige/mcc4mcc.svg?branch=master)](https://travis-ci.org/cui-unige/mcc4mcc)

# Model Checker Collection for the Model Checking Contest @ Petri nets

The model checker collection is a tool that is built to compete in the
[Model Checking Contest @ Petri nets](https://mcc.lip6.fr)
using the tools that are already competing.

Its purpose is to allow research on the algorithm that will choose
the best tool, depending on the model, examination and formula characteristics.

# Running the tool

Help can be obtained through:
```sh
$ python -m mcc --help
```

The following command extracts known and learned data from the results
of the 2017 edition of the Model Checking Contest:

```sh
$ python -m mcc extract --year=2017 --duplicates
```

The following command tests if docker images can be run on some examinations
and models of the Model Checking Contest:

```sh
$ python -m mcc test --year=2017
```

The following command runs `mcc4mcc` with the state space examination
on the model stored in `./models/TokenRing-PT-005.tgz`:

```sh
$ python -m mcc run \
    --examination=StateSpace \
    --input=./models/TokenRing-PT-005.tgz
```

Models can be obtained in the
[MCC Submission Kit](https://mcc.lip6.fr/archives/ToolSubmissionKit.tar.gz).

## Forgetting some model characteristics

In order to create more collisions between models given a set of
characteristics, it can be interesting to forget some characteristics
during machine learning.
The `--forget` option allows is, for instance:

```sh
$ python -m mcc extract \
    --duplicates \
    --year=2017 \
    --forget "Deadlock,Live,Quasi Live,Reversible,Safe"
```


# Obtaining the tool submission kit and models

The tool submission kit can be downloaded automatically,
and models extracted from it using the following command:

```sh
$ ./prepare
```

# Building the docker images

This repository provides scripts to build automatically the docker images
using virtual machines of the previous edition of the Model Checking Contest.
This is not optimal: tool developers should provide their own docker images
for both simplicity and efficiency.

The following command builds all images and uploads them to the docker
image repository `mccpetrinets`:

```sh
$ ./build
```

# Building the virtual machine

The following command creates a virtual machine containing `mcc4mcc`
dedicated to the Model Checking Contest:

```sh
$ ./install
```

The virtual machine is created as `mcc4mcc-2018.vmdk`.
