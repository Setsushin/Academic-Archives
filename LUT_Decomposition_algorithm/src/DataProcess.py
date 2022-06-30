import numpy as np
import random as rd

# def producer(num, mode = 'random'):  
#     numlist = []
#     for i in range((2**num)):
#         n1 = str(bin(i))  #string
#         n1 = n1[2:len(n1)]  #delete '0x'
#         n1 = n1.zfill(num)   #fill zeros

#         n2 = str(bin(rd.randint(0,2**(num-1))))  #string
#         n2 = n2[2:len(n2)]  #delete '0x'
#         n2 = n2.zfill(num)   #fill zeros
        
#         numlist.append((n1, n2))
#     return numlist

def multiplier_producer(Bit, Datatype):  
    numlist = []
    for i in range(2**(Bit//2)):
        for j in range(2**(Bit//2)):
            output = i*j
            b_input = str(bin(i))[2:].zfill(Bit//2) + str(bin(j))[2:].zfill(Bit//2)
            b_output = str(bin(output))[2:].zfill(Bit)
            numlist.append((b_input, b_output))
    return numlist
    
def generate_txt(data):
    f1 = open('data.txt', 'w')
    for i in data:
        for j in i:
            f1.write(j)
            f1.write(' ')
        f1.write('\n')
    f1.close()