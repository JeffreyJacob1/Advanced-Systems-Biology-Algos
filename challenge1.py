import sys
import numpy as np

"""The following uses Python to challenge you to create an algorithm for finding
matches between a set of aligned strings. Minimal familiarity with Python is 
necessary, notably list and Numpy array slicing. 
"""

"""Problem 1.

Let X be a list of M binary strings (over the alphabet { 0, 1 }) each of length 
N. 

For integer 0<=i<=N we define an ith prefix sort as a lexicographic sort 
(here 0 precedes 1) of the set of ith prefixes: { x[:i] | x in X }.
Similarly an ith reverse prefix sort is a lexicographic sort of the set of
ith prefixes after each prefix is reversed.

Let A be an Mx(N+1) matrix such that for all 0<=i<M, 0<=j<=N, A[i,j] is the 
index in X of the ith string ordered by jth reverse prefix. To break ties 
(equal prefixes) the ordering of the strings in X is used. 

Complete code for the following function that computes A for a given X.

Here X is a Python list of Python strings. 
To represent A we use a 2D Numpy integer array.

Example:

>>> X = getRandomX() #This is in the challenge1UnitTest.py file
>>> X
['110', '000', '001', '010', '100', '001', '100'] #Binary strings, M=7 and N=3
>>> A = constructReversePrefixSortMatrix(X)
>>> A
array([[0, 1, 1, 1],
       [1, 2, 2, 4],
       [2, 3, 5, 6],
       [3, 5, 4, 3],
       [4, 0, 6, 0],
       [5, 4, 3, 2],
       [6, 6, 0, 5]])
>>> 

Hint:
Column j (0 < j <= N) of the matrix can be constructed from column j-1 and the 
symbol in each sequence at index j-1.  

Question 1: In terms of M and N what is the asymptotic cost of your algorithm?

#The asymptotic cost of this function is O(M N^2), where M is the number of binary strings in the input list X, and N is the length of each binary string.
#the reason for this is that the function sorts M binary strings of length N, which has a cost of O(M N log N),
and also performs a nested loop over M and N which has a cost of O(M N)

"""


def constructReversePrefixSortMatrix(X):
    M = len(X)
    N = len(X[0]) if M > 0 else 0
    # Create an empty Numpy array to store the results
    A = np.empty(shape=[M, N+1], dtype=int) 
    
    # Loop through each column of the result matrix (0 <= j <= N)
    for j in range(N+1):
        # Sort the jth prefixes of each binary string in X by reverse prefix order
        k = sorted([(X[i][:j][::-1], i) for i in range(M)]) 
        # Fill in the jth column of the result matrix with the indices of the sorted binary strings
        A[:,j] = [i[1] for i in k]
    
    return A











"""Problem 2: 

Following on from the previous problem, let Y be the MxN matrix such that for 
all 0 <= i < M, 0 <= j < N, Y[i,j] = X[A[i,j]][j].

Complete the following to construct Y for X. 

Hint: You can either use your solution to constructReversePrefixSortMatrix() 
or adapt the code from that algorithm to create Y without using 
constructReversePrefixSortMatrix().





Question 2: In terms of M and N what is the asymptotic cost of your algorithm?
"""
'''
The asymptotic cost of the algorithm would be O(M * N log N) because for
for each column j in the A matrix, the sorting of the (X[i][:j][::-1], i)
or i in range(M) takes O(M log M) time for the built in sorted funtion, and there are N+1 columns to be sorted,
resulting in a total time complexity of O(M * N log N).
'''





def constructYFromX(X):
    M = len(X)
    N = len(X[0]) if M > 0 else 0
    # A is a matrix of M rows and N+1 columns to store the reverse prefix sorted indices of X
    A = np.empty(shape=[M, N+1], dtype=int) 
    for j in range(N+1):
        k = [(X[i][:j][::-1], i) for i in range(M)] # create a list of tuples, where each tuple consists of the reversed suffix of X[i][:j] and the index i
        k = sorted(k) # sort the list of tuples based on the reversed suffix
        # store the indices of X in the j-th column of A
        A[:,j] = [i[1] for i in k]
    
    Y = np.empty(shape=[M, N], dtype=int)
    for i in range(M):
        for j in range(N):
            # set the j-th character of the i-th string of Y to be the j-th character of X[A[i][j]]
            Y[i][j] = X[A[i][j]][j]
            
    return Y




"""Problem 3.

Y is a transformation of X. Complete the following to construct X from Y, 
returning X as a list of strings as defined in problem 1.
Hint: This is the inverse of X to Y, but the code may look very similar.

Question 3a: In terms of M and N what is the asymptotic cost of your algorithm?

#the asymtotic cost is O(MN) where is the is rows and y is the collums of Y. we loop through each to construct A.

Question 3b: What could you use the transformation of Y for? 
Hint: consider the BWT.
#transforming y to x could be used to compute the BWT of a string, transformation of y can be used to determine A and recreate x
Question 3c: Can you come up with a more efficient data structure for storing Y?

#using a suffix tree would make the answer more efficient as they are efficient at string matching and manipulation
a suffix tree could be used to store the rows of Y in a way that makes it efficient to find the next row
 to add to the current column of A.
"""


def constructXFromY(Y):
    x, y = Y.shape
    A = np.empty(shape=[x, 0 if x == 0 else y], dtype=int)
    # initialize the list curr_col with elements 0, 1, ..., x - 1
    curr_col = list(range(x))
    for i in range(y):
        # fill the current column of A with the values from the corresponding column of Y
        for j in range(x):
            A[curr_col[j], i] = Y[j, i]
        # create two lists, zeros and ones, containing the indices of curr_col corresponding to the rows in Y with 0's and 1's, respectively
        zeros = [curr_col[k] for k in range(x) if Y[k, i] == 0]
        ones = [curr_col[k] for k in range(x) if Y[k, i] == 1]
        # update curr_col to be the concatenation of zeros and ones
        curr_col = zeros + ones
    # convert the numpy array A to a list of strings X, where each string represents a row of A
    X = [''.join(map(str, row)) for row in A.tolist()]
    return X





"""Problem 4.

Define the common suffix of two strings to be the maximum length suffix shared 
by both strings, e.g. for "10110" and "10010" the common suffix is "10" because 
both end with "10" but not both "110" or both "010". 

Let D be a Mx(N+1) Numpy integer array such that for all 1<=i<M, 1<=j<=N, 
D[i,j] is the length of the common suffix between the substrings X[A[i,j]][:j] 
and X[A[i-1,j]][:j].  

Complete code for the following function that computes D for a given A.

Example:

>>> X = getRandomX()
>>> X
['110', '000', '001', '010', '100', '001', '100']
>>> A = constructReversePrefixSortMatrix(X)
>>> A
array([[0, 1, 1, 1],
       [1, 2, 2, 4],
       [2, 3, 5, 6],
       [3, 5, 4, 3],
       [4, 0, 6, 0],
       [5, 4, 3, 2],
       [6, 6, 0, 5]])
>>> D = constructCommonSuffixMatrix(A, X)
>>> D
array([[0, 0, 0, 0],
       [0, 1, 2, 2],
       [0, 1, 2, 3],
       [0, 1, 1, 1],
       [0, 0, 2, 2],
       [0, 1, 0, 0],
       [0, 1, 1, 3]])

Hints: 

As before, column j (0 < j <= N) of the matrix can be constructed from column j-1 
and thesymbol in each sequence at index j-1.

For an efficient algorithm consider that the length of the common suffix 
between X[A[i,j]][:j] and X[A[i-k,j]][:j], for all 0<k<=i is 
min(D[i-k+1,j], D[i-k+2,j], ..., D[i,j]).

Question 4: In terms of M and N what is the asymptotic cost of your algorithm?
"""
# O(MN) is the asymtotic cost as there is two nested loops with the outer loop running M times and the
#inner loop running N times. This results in a total of M * N






def constructCommonSuffixMatrix(A, X):
    """
    Constructs the common suffix matrix for given A and X
    """
    # Create the Mx(N+1) D matrix
    rows, cols = A.shape
    D = np.zeros(shape=(rows, cols), dtype=int)

    # Set the first column to all 0's
    D[:, 0] = 0

    # Determine the length of the first for loop
    length = len(X[0]) if rows > 0 else 0

    for i in range(length):
        zeros = [] # list to store the suffix length of 0's
        ones = [] # list to store the suffix length of 1's
        p = q = 0 # initialize p and q as 0

        for j in range(rows):
            p = min(p, D[j, i] + 1) # find the minimum between p and (D[j, i] + 1)
            q = min(q, D[j, i] + 1) # find the minimum between q and (D[j, i] + 1)
            index = X[A[j, i]][i] # get the index from X based on A[j, i]

            if index == "0": # if the index is 0
                zeros.append(p) 
                p = sys.maxsize # reset p to sys.maxsize
            elif index == "1": # if the index is 1
                ones.append(q) 
                q = sys.maxsize # reset q to sys.maxsize
        
        curr_range = zeros + ones # create the range by combining the zeros and ones list
        D[:, i + 1] = curr_range # set the values in the current column to curr_range
    
    return D




"""Problem 5.
    
For a pair of strings X[x], X[y], a long match ending at j is a common substring
of X[x] and X[y] that ends at j (so that X[x][j] != X[y][j] or j == N) that is longer
than a threshold 'minLength'. E.g. for strings "0010100" and "1110111" and length
threshold 2 (or 3) there is a long match "101" ending at 5.
    
The following algorithm enumerates for all long matches between all substrings of
X, except for simplicity those long matches that are not terminated at
the end of the strings.
    
Question 5a: What is the asymptotic cost of the algorithm in terms of M, N and the
number of long matches?
  O(MN*longMatches)   
Question 5b: Can you see any major time efficiencies that could be gained by
refactoring?
    changing the code to have less iterations could help with the time efficiencies
Question 5c: Can you see any major space efficiencies that could be gained by
refactoring?

There could potentially be space efficiencies gained by refactoring the algorithm by using more efficient data structures 
to store the reverse prefix sort matrix and common suffix matrix, as well as reducing the number
of working arrays used to store the indices of strings containing long matches.
 
Question 5d: Can you imagine alternative algorithms to compute such matches?,
if so, what would be the asymptotic cost and space usage?

Alternative algorithms to compute long matches could include using hashing techniques, such as the Rabin-Karp algorithm,
which would have an asymptotic cost of O(M * N) and would have lower space complexity.

"""
def getLongMatches(X, minLength):
    assert minLength > 0
    
    A = constructReversePrefixSortMatrix(X)
    D = constructCommonSuffixMatrix(A, X)
    
    #For each column, in ascending order of column index
    for j in range(1, 0 if len(X) == 0 else len(X[0])):
        #Working arrays used to store indices of strings containing long matches
        #b is an array of strings that have a '0' at position j
        #c is an array of strings that have a '1' at position j
        #When reporting long matches we'll report all pairs of indices in b X c,
        #as these are the long matches that end at j.
        b, c = [], []
        
        #Iterate over the aligned symbols in column j in reverse prefix order
        for i in range(len(X)):
            #For each string in the order check if there is a long match between
            #it and the previous string.
            #If there isn't a long match then this implies that there can
            #be no long matches ending at j between sequences indices in A[:i,j]
            #and sequence indices in A[i:,j], thus we report all long matches
            #found so far and empty the arrays storing long matches.
            if D[i,j] < minLength:
                for x in b:
                    for y in c:
                        #The yield keyword converts the function into a
                        #generator - alternatively we could just to append to
                        #a list and return the list
                        
                        #We return the match as tuple of two sequence
                        #indices (ordered by order in X) and coordinate at which
                        #the match ends
                        yield (x, y, j) if x < y else (y, x, j)
                b, c = [], []
            
            #Partition the sequences by if they have '0' or '1' at position j.
            if X[A[i,j]][j] == '0':
                b.append(A[i,j])
            else:
                c.append(A[i,j])
        
        #Report any leftover long matches for the column
        for x in b:
            for y in c:
                yield (x, y, j) if x < y else (y, x, j)
