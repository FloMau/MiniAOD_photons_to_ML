#!/bin/bash

source /net/software/deep-learning/virtualenv/python3.9/tensorflow2.10/bin/activate
cd ~/work
python3 vit_search.py models/vit_base.json
