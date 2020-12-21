#!/usr/bin/env python
'''
From D. Noonan, copied exactly on Jun 18, 2020 from
https://github.com/dnoonan08/HGCAL_Scripts/blob/master/VerificationData/encode.py
'''
import numpy as np
import math

def encode(value, dropBits=1, expBits=4, mantBits=3, roundBits=False, asInt=False):
    
    binCode=bin(value)[2:]
    
    if len(binCode) <= (mantBits+dropBits):
        if roundBits and dropBits>0:
            value += 2**(dropBits-1)
        value = value>>dropBits
        binCode=bin(value)[2:]
        
        mantissa = format(value, '#0%ib'%(mantBits+2))[2:]
        exponent = '0'*expBits
    elif len(binCode)==mantBits+dropBits+1:
        if roundBits and dropBits>0:
            value += 2**(dropBits-1)
        value = value>>dropBits
        binCode=bin(value)[2:]
        exponent = '0001'
        mantissa = binCode[1:1+mantBits]
    else:
        if roundBits:
            vTemp = int(binCode,2) + int(2**(len(binCode)-2-mantBits))
            binCode = bin(vTemp)[2:]
        firstZero = len(binCode)-mantBits-dropBits
        if firstZero<1:
            print ("PROBLEM")
        if firstZero<2**expBits:
            exponent = format(firstZero, '#0%ib'%(expBits+2))[2:]
            mantissa = binCode[1:1+mantBits]

        else:
            exponent = '1'*expBits
            mantissa = '1'*mantBits
            
    if asInt:
        return int(exponent + mantissa,2)
    else:
        return exponent + mantissa
        
def decode(val,droppedBits=1,expBits=4,mantBits=3,edge=False,quarter=False):

    exp=val>>mantBits
    mant= val & (2**mantBits-1)

    data = (((mant<<(exp-1)) if exp>0 else mant) + (0 if exp==0 else (1<<(exp+mantBits-1))))
    data = data<<droppedBits

    shift = max(exp-1,0)
    if quarter:
        if (droppedBits+shift)>1:
            data += 1<<(shift+droppedBits-2)
    elif not edge:
        if (droppedBits+shift)>0:
            data += 1<<(shift+droppedBits-1)
    return data
