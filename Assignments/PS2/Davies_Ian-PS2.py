import numpy as np
import pandas as pd
import scipy as sp
from scipy import sparse
from scipy.sparse import csc_matrix
import time

start_time = time.clock() # start timing the program

# constants
alpha = 0.85
epsilon = 0.00001

# read data
dat = pd.read_csv("links.txt") # reading as df is faster than reading as array
dat = dat.as_matrix() # convert to dense matrix

# create empty adjacency matrix
n = len(np.unique(dat[:,1])) # each individual journal will occupy one columna and one row
Z = np.zeros((n,n)) # n x n zero matrix

# fill empty matrix Z with citations
for i in range(0, dat.shape[0]-1): # journals are numbered sequentially, so journal 1 will be indexed as [1,1] in our matrix
    Z[dat[i,1], dat[i,0]] = dat[i,2]

# set diagonals to zero
np.fill_diagonal(Z,0)

# normalize columns
H = np.divide(Z, Z.sum(axis=0)) # each column now sums to 1
H = np.nan_to_num(H) # set NaN to 0 (artifact of some columns having sum = 0)

# identify dangling nodes by adding row vector d 
# d=1 if journal doesn't cite other journals, d=0 if it does
d = np.zeros((len(H))) # empty vector
for i in range(0,len(H)-1): # if journal has no outgoing citations, 1, else 0
    if Z[:,i].sum(axis=0) == 0:
        d[i] = 1
    else:
        d[i] = 0
        
# create article vector
A_tot = len(H) # assume all journals publish one article
a = np.ones((1, len(H))) 
a = a.reshape((len(H),1)) # reshape from row to column vector
a = np.divide(a, A_tot) # normalize so vector sums to 1

# initial start vector
pi0 = np.zeros((len(Z),1)) # start pi0 at 0 so that L1 norm of pi-pi0 is > epsilon initially
pi = (np.ones((len(Z), 1))) / len(Z) # start pi for all journals equally, 1/n
count=1 # start iteration counter

H = sp.sparse.csr_matrix(H) # convert H to sparse matrix to conserve memory

# calculate influence vector
# while np.linalg.norm((pi-pi0),ord=1) > epsilon:
while np.sum(np.abs(pi-pi0)) > epsilon: # iterate until L1 norm is <= epsilon
    count+=1
    pi0 = pi.copy() # pi0 is the previous iteration of pi
    pi = (alpha * H * pi0) + (alpha * np.dot(d, pi0) + (1-alpha))*a # calculate new iteration 
print(count)

# calculate eigenfactor
EF = 100* (sparse.csr_matrix.dot(H, pi)/sum(sparse.csr_matrix.dot(H, pi)))

# print time it took to run program
print ("{:.2f}".format(time.clock() - start_time), "seconds") 

# top 20 journals (left) and scores (right)

np.set_printoptions(suppress=True)
np.hstack((np.argsort(-EF, axis=0)[0:19],
         sorted(EF, reverse=True)[0:19]))

# Time it took to run = 17.17 seconds
# Iterations = 33
# Journals (left) and top scores (right)
#      ([[4408, 1.44811939],
#        [4801, 1.41271931],
#        [6610, 1.23503493],
#        [2056, 0.67950245],
#        [6919, 0.66487922],
#        [6667, 0.63463528],
#        [4024, 0.57723329],
#        [6523, 0.48081521],
#        [8930, 0.47777283],
#        [6857, 0.43973514],
#        [5966, 0.42971781],
#        [1995, 0.38620683],
#        [1935, 0.38512031],
#        [3480, 0.37957765],
#        [4598, 0.3727892 ],
#        [2880, 0.33030656],
#        [3314, 0.32750816],
#        [6569, 0.3192717 ],
#        [5035, 0.31677921]])
