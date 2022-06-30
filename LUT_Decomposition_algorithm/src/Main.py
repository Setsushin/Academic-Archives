# MLUT
# Given input&output data(up to 16-bit), decide the optimization space

import argparse
import os
import numpy as np
import random as rd
import itertools
import time

# 自己写的定制功能
from DataProcess import multiplier_producer, generate_txt
from utils import binarize, count_outputs, calculate_loss, find_best_loss, find_dividing_line

# Args(Hyper Parameters)
parser = argparse.ArgumentParser(description='DataProcess')
parser.add_argument('--DataType', default = 'Multiplier',
                    help='Data type of data to be generated')
parser.add_argument('--Bit', default = 8,
                    help='Bit amount of data to be generated')
parser.add_argument('-MER', default = 0.3,
                    help='Max Error Rate allowed in approximation')
args = parser.parse_args()


# Generate Data
if args.DataType == 'Multiplier':
    data = multiplier_producer(args.Bit, args.DataType)
    generate_txt(data)
else:
    print('Not supported data type.')
    os._exit()

# Load Data
data = np.loadtxt('data.txt')
data_length = data.shape[0]
input_bit = output_bit = args.Bit

input_data = data[:,0]
output_data = data[:,1]

# B = Total input bit, M = reserved bit, N = optimized bit
B = input_bit
M = 0
N = B - M
    

if __name__ == '__main__':
    start = time.time()
    sorted_output_data = sorted(output_data)
    # 记录输出值的分布情况
    record_of_output = count_outputs(sorted_output_data, data_length)
    # 计算结果
    result = find_best_loss(record_of_output, sorted_output_data, N)

    best_no_loss_optim = -1

    for i in range(len(result)):
        if result[i][1] == 0:
            best_no_loss_optim = i

    # print(result)
    result = result[::-1]

    for i in range(len(result)):
        if result[i][1] != 0:
            best_reduced_bit = i+1
            best_correct = result[i][0]
            best_loss = result[i][1]
    
    # Print out results    
    print('')
    print('The original LUT size: ({},{})'.format(input_bit, input_bit))
    print('It has {} non-repeated output individuals'.format(len(record_of_output)))
    print('')
    print('The amount of reserved bit: {}'.format(M))
    print('BEST RESULTS with no loss:')
    print('Optimized LUT: ({},{}) + ({},{})'.format(input_bit, input_bit - best_no_loss_optim, input_bit - best_no_loss_optim, input_bit))
    # if M != 0:
    #     print('The {} reversed bits are:'.format(M))
    print('')
    print('BEST RESULTS with loss:')
    print('Optimized LUT: ({},{}) + ({},{})'.format(input_bit, best_reduced_bit, best_reduced_bit, input_bit))
    print('Correct mapping pairs: {}/{}. The final loss is {:.4f}'.format(best_correct, best_correct + best_loss, best_loss/(best_correct + best_loss)))
    print('')

    bi_input_data = binarize(input_data, args.Bit)
    bi_output_data = binarize(output_data, args.Bit)
    bi_data = []
    for i in range(len(bi_input_data)):
        bi_data.append([bi_input_data[i], bi_output_data[i]])
    
    while M < input_bit//2:
        space = np.arange(B) # [0,1,2,3,4,5,6,7]
        M = M + 1 # M = 1
        N = B - M # N = 7
        reserved_bit_choice = list(itertools.combinations(space, M))
        
        best_global = []
        best_correct_global, best_loss_global, best_reduced_bit_global, best_choice_global = 2**64,2**64,2**64,2**64
        
        for current_choice in reserved_bit_choice:
            bi_data_splited = []
            for i in bi_data:
                reserved_bit = ''
                last_bit = ''
                count = 0
                for j in i[0]:
                    if count in current_choice:
                        reserved_bit += j
                    else:
                        last_bit += j
                    count += 1
                bi_data_splited.append([reserved_bit, last_bit, i[1]])

            bi_data_splited = sorted(bi_data_splited, key=lambda x:x[0]) # sorted by reserved-bit

            dividing_line = find_dividing_line(np.array(bi_data_splited)[:,0]) # find the dividing line of reserved-bit
            dividing_line.append(len(bi_data_splited)-1)

            # print(bi_data_splited)
            # print(dividing_line)

            outset = 0
            result_storage = []
            for i in dividing_line:
                temp = bi_data_splited[outset:(i+1)] # cut outset~i
                temp = sorted(temp, key=lambda x:x[-1]) # sort by output values
                temp_output = np.array(temp)[:,-1] # output column
                # print(temp)

                ## process templist
                record_of_output = count_outputs(temp_output, len(temp_output))
                result = find_best_loss(record_of_output, temp_output, N)
                result_storage.append(result)
                ## process templist

                outset = i+1
            
            best_reduced_bit, best_correct, best_loss = -1, -1, -1
            
            # print(result_storage)
            
            for i in range(N):
                a,b = 0,0
                for j in result_storage:
                    a += j[i][0]
                    b += j[i][1]
                if b/(a+b) < args.MER:
                    best_correct = a
                    best_loss = b
                    best_reduced_bit = i
            
            if best_loss < best_loss_global and best_loss > 0:
                best_correct_global = best_correct
                best_loss_global = best_loss
                best_reduced_bit_global = best_reduced_bit
                best_choice_global = current_choice

            # Print out results
            print('')
            print('The {} reserved bit: No.{}'.format(M, current_choice))

            # *** don't need no loss ****
            # if best_no_loss_optim > 0:
            #     print('Best Results with no loss:')
            #     print('Optimized LUT is ({},{}) + ({},{})'.format(N, best_no_loss_optim, best_no_loss_optim, output_bit))
            # else:
            #     print('No Best Results without loss.')

            if best_reduced_bit > 0:
                print('Best Results with loss:')
                print('Optimized LUT: ({},{}) + ({},{})'.format(N, N - best_reduced_bit, input_bit - best_reduced_bit, input_bit))
                print('Correct mapping pairs: {}/{}. The final loss is {:.4f}'.format(best_correct, best_correct + best_loss, best_loss/(best_correct + best_loss)))
            else:
                print('No Best Results with loss (over max loss limit).')
            
            print('')
            
        best_global.append((M, best_choice_global, best_correct_global, best_loss_global, best_reduced_bit_global))

        # print(best_global)

        # Print Global optimization result
        print('Best Global Results:')
        for i in best_global:
            if i[1] != 2**64:
                print('{} Reserved bit: {}, Correct mapping pairs: {}/{}. Final loss is {:.4f}'.format(i[0], i[1], i[2], i[2]+i[3], i[3]/(i[2]+i[3]) ))
            else:
                print('{} Reserved bit: no optimal results.'.format(i[0]))
        
        end = time.time()
        print('operation time:', end - start)