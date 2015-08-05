#!/usr/bin/sh

import math
import os

src = "l1.png"
for i in range(2,50):
    per = int(math.sqrt(i)*100)
    dst = "l" + str(i) + ".png"
    os.system("convert {0} -resize {1}% {2}".format(src,per,dst))
