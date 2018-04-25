[![Build Status](https://travis-ci.org/cui-unige/mcc4mcc.svg?branch=master)](https://travis-ci.org/cui-unige/mcc4mcc)

# Model Checker Collection for the Model Checking Contest @ Petri nets

The model checker collection is a tool that is built to compete in the
[Model Checking Contest @ Petri nets](https://mcc.lip6.fr)
using the tools that are already competing.

Its purpose is to allow research on the algorithm that will choose
the best tool, depending on the model, examination and formula characteristics.

## Installing the tool

This tool can be installed easily with `pip` from the sources:

```sh
$    git clone https://github.com/cui-unige/mcc4mcc.git \
  && cd mcc4mcc \
  && pip install .
```
We currently do not distribute a packaged version.

## Obtaining the tool submission kit and models

The [MCC Submission Kit](https://mcc.lip6.fr/archives/ToolSubmissionKit.tar.gz)
can be downloaded automatically, and models extracted from it using the
following command:

```sh
$ ./prepare
```

The submission kit is put in the `ToolSubmissionKit` directory,
and models are copied in the `models` directory.

## Running the tool

Help can be obtained through:

```sh
$ python3 -m mcc4mcc \
    --help
```

## Extracting information from the previous edition

The following command extracts known and learned data from the results
of the 2017 edition of the Model Checking Contest:

```sh
$ python3 -m mcc4mcc \
    extract \
    --year=2017
```

It creates several files, that are used to chose the correct tool to run:

* `<prefix>-configuration.json`
* `<prefix>-known.json`
* `<prefix>-learned.json`
* `<prefix>-learned.<algorithm>.p`
* `<prefix>-values.json`

## Running the model checker collection

The following command runs `mcc4mcc` with the state space examination
on the model stored in `./models/TokenRing-PT-005.tgz`.
The `prefix` option tells the tool to use files generated with
the given prefix.
It allows users to generate files for several extraction parameters,
and use them by giving their prefix.

```sh
$ python3 -m mcc4mcc \
    run \
    --examination=StateSpace \
    --input=./models/TokenRing-PT-005.tgz \
    --prefix=7e556e9247727f60
```

## Testing the docker images

The following command tests if docker images can be run on some examinations
and models of the Model Checking Contest:

```sh
$ python3 -m mcc4mcc \
    test \
    --year=2017
```

## Forgetting some model characteristics

In order to create more collisions between models given a set of
characteristics, it can be interesting to forget some characteristics
during machine learning.
The `--forget` option allows us to do it, for instance:

```sh
$ python3 -m mcc4mcc extract \
    --duplicates \
    --year=2017 \
    --forget="Deadlock,Live,Quasi Live,Reversible,Safe"
```

## Dropping some models during machine learning

It is interesting to remove some models during machine learning,
in order to check that the algorithm is still able to obtain a good score.
The `--training` option allows us to do it.

```sh
$ python3 -m mcc4mcc extract \
    --duplicates \
    --year=2017 \
    --training=0.25
```

The command above keeps only 25% of the models for learning,
but still computes the score using all the models.
The `--duplicates` option allows the tool to keep duplicate lines
during machine learning.

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
$ ./create-vm
```

The virtual machine is created as `mcc4mcc.vmdk`.
