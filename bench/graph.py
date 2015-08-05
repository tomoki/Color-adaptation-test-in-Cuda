#!/usr/bin/env python

import subprocess
if "check_output" not in dir( subprocess ): # duck punch it in!
    def f(*popenargs, **kwargs):
        if 'stdout' in kwargs:
            raise ValueError('stdout argument not allowed, it will be overridden.')
        process = subprocess.Popen(stdout=subprocess.PIPE, *popenargs, **kwargs)
        output, unused_err = process.communicate()
        retcode = process.poll()
        if retcode:
            cmd = kwargs.get("args")
            if cmd is None:
                cmd = popenargs[0]
            raise subprocess.CalledProcessError(retcode, cmd)
        return output
    subprocess.check_output = f

def input_file(i):
    return "res/l" + str(i) + ".png"

def gen_data():
    N = 50
    r = []
    for i in range(1,N):
        here = input_file(i)
        k = [subprocess.check_output(("../invert {0} /tmp/out.png {1}".format(here, str(j))).split(" ")).strip() for j in range(3)]
        k[:0] = [str(i)] # prepend
        r.append(" ".join(k))
    return "\n".join(r)

print(gen_data())
