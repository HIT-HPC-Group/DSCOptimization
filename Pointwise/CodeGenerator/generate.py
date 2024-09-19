import os
import time

with open("parameters.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        cli = "./codegen" + " " + line
        os.system(cli)