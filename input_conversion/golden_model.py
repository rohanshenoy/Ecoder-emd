#!/usr/bin/env python
'''
Golden model for the trigger cell charge sum calculation.
Integer TC charges are read, restricted to 22b uints
and summed to a 22b uint. The sum is encoded using Danny's
functions defined in the accompanying file 'encode.py'
C. Herwig
'''

import numpy as np
from encode import encode,decode

def bound(x, max_Q_bits = 22):
    if x<0: return 0
    if x >= 2**max_Q_bits:
        x = bound  % (2**max_Q_bits)
    return x

def main():
 
    test_data = np.genfromtxt('../sim/tb_data/tb_input.dat',
                          delimiter=' ', dtype=int)
    
    nDebug=3
    ind=0
    with open('tb_encoded_sum.dat','w') as f:
        for tcs in test_data:
            sumQ = bound(np.sum(list(map(bound,tcs))))
            encoded = encode(sumQ, expBits=5, mantBits=4, dropBits=0)
            f.write( encoded + '\n')
            if ind<nDebug:
                print("{} sum is {}, encoding {}".format(ind, sumQ, encoded))
            ind += 1
        
if __name__ == "__main__":
    main()
