import csv
import numpy as np
import math

input_file_1 = '0_example_7500x50.csv'
input_file_2 = '0_example_75000x150.csv'

def main():
    perceptron(input_file_2, 'percept_output_1.csv')
    mw_hedge(input_file_2, 'mw_output_1.csv',math.log(1.05))


def sign(n):
    if n < 0:       # returns the alg's prediction
        return -1   # based on the sign of 
    else:           # the dot product
        return 1

def perceptron(input_file, output_file):
    input = open(input_file)
    input_reader = csv.reader(input, delimiter=',')

    output = open(output_file, "w",newline='')
    perc_writer = csv.writer(output, delimiter=',')

    row1 = next(input_reader)

    w = np.zeros(len(row1)-1)

    example_num = 0
    perc_writer.writerow(np.around(np.concatenate((np.array([example_num]),w)),5))

    for row in input_reader:
        x = np.array(row[1:len(row)],dtype=float) # example
        label = int(row[0]) # its given label
        p = sign(np.dot(w,x)) # alg's prediction for the example
        example_num += 1
        if p == label: # check if we predicted correctly
            continue
        else: # if not, correct ourselves
            w = w + label * x
            perc_writer.writerow(np.around(np.concatenate((np.array([example_num]),w)),5))
    
    
    input.close()
    output.close()


'''
returns the alg's prediction based on 
the total weights of the experts which 
predict 1 vs those who predict -1
'''
def mw_predict(w,experts):
    w_n = 0
    w_p = 0
    for i in range(len(w)):
        if experts[i] == 1:
            w_p += w[i]
        else:
            w_n += w[i]

    return 1 if w_p >= w_n else -1

def mw_hedge(input_file, output_file, n):
    input = open(input_file)
    input_reader = csv.reader(input, delimiter=',')

    output = open(output_file, "w",newline='')
    mw_writer = csv.writer(output, delimiter=',')

    row1 = next(input_reader)
    num_experts = len(row1)-1
    w = np.ones(num_experts)/num_experts # initialize equal weights

    example_num = 0
    mw_writer.writerow(np.around(np.concatenate((np.array([example_num]),w)),5))

    for row in input_reader:
        experts = np.array(row[1:len(row)],dtype=float)
        label = int(row[0])

        p = mw_predict(w,experts)
        example_num += 1
        if p == label:
            continue
        else:
            w = w*np.exp(n*label*experts)
            mw_writer.writerow(np.around(np.concatenate((np.array([example_num]),w)),5))
        
    input.close()
    output.close()
    
if __name__ == "__main__":
    main()
