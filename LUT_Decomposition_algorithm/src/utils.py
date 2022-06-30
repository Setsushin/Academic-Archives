import numpy as np
import random as rd

# Binarize
def binarize(data, bit):
    numlist = []
    for i in data:
        n = str(int(i)) #string
        n = n.zfill(bit)   #fill zeros
        numlist.append(n)
    return numlist

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

# calculate the loss
def calculate_loss(data, record, bit):
    sorted_record = sorted(record, key=lambda x:x[1], reverse=True) # Descending order
    num_of_outputs = len(data)
    rest_output_individuals = 0
    for index in range(2**bit):
        rest_output_individuals += sorted_record[index][1]
    return rest_output_individuals, num_of_outputs - rest_output_individuals

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