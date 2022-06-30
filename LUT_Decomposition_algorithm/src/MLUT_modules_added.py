# MLUT
# Given input&output data(up to 16-bit), decide the optimization space

import argparse
import os
import numpy as np
import random as rd
import itertools

# Hyper Parameters
BIT = 16
Max_individual_loss = 0.3
# Max_category_loss = 0.3

# Load Data
data = np.loadtxt('data.txt')
data_length = data.shape[0]
input_bit = BIT
output_bit = BIT

input_data = data[:,0]
output_data = data[:,1]

# B = Total input bit, M = reserved bit, N = optimized bit
B = input_bit
M = 0
N = B - M 

# Funtions
def binarize(data, bit=input_bit):
    numlist = []
    for i in data:
        n = str(int(i)) #string
        n = n.zfill(bit)   #fill zeros
        numlist.append(n)
    return numlist

# calculate the loss
def calculate_loss(data, record, bit):
    sorted_record = sorted(record, key=lambda x:x[1], reverse=True) # Descending order
    num_of_outputs = len(data)
    rest_output_individuals = 0
    for index in range(2**bit):
        rest_output_individuals += sorted_record[index][1]
    return rest_output_individuals, num_of_outputs - rest_output_individuals


# count non-repeat output individuals and record in list
def count_outputs(sorted_output_data, data_length):
    record_of_output = []
    num_of_current_outputs = 1

    for index in range(data_length - 1):
        if sorted_output_data[index + 1] != sorted_output_data[index]:
            record_of_output.append((int(sorted_output_data[index]), num_of_current_outputs))
            num_of_current_outputs = 1
        else:
            num_of_current_outputs += 1
    record_of_output.append((int(sorted_output_data[-1]), num_of_current_outputs))

    return record_of_output

# find loss
def find_best_loss(record_of_output, data, N):
    current_total_bit = N
    output_individuals = len(record_of_output)
    result = []

    while current_total_bit >= 1:
        if output_individuals <= 2**current_total_bit:
            result.append((len(data), 0))
            current_total_bit -= 1
        else:
            correct, loss = calculate_loss(data = data, record = record_of_output, bit = current_total_bit)
            result.append((correct, loss))
            current_total_bit -= 1
            
    return result

def find_dividing_line(data):
    dividing_line = []
    for index in range(len(data) - 1):
        if data[index] != data[index + 1]:
            dividing_line.append(index)
    return dividing_line
    

if __name__ == '__main__':
    sorted_output_data = sorted(output_data)
    record_of_output = count_outputs(sorted_output_data, data_length)
    result = find_best_loss(record_of_output, sorted_output_data, N)

    print(len(record_of_output))
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

    bi_input_data = binarize(input_data)
    bi_output_data = binarize(output_data)
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
                if b/(a+b) < Max_individual_loss:
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