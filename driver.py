# coding=utf-8
import os

os.system('python ./main.py evaluate --model ./workDir/trainResult30/my-model-ner-30 --test ./workDir/data/testdata'
          ' --score_dir ./workDir/trainResult30')