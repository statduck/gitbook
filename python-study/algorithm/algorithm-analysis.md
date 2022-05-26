# Algorithm Analysis

## Big-Oh Notation

$$
f(n) \leq cg(n), \quad for \; n\leq n_0
$$

It is called as "f(n) is **big-Oh** of g(n)".&#x20;

![](<../../.gitbook/assets/image (48).png>)

For example

* $$8n+5$$ is $$O(n)$$
* $$5n^4+3n^3+2n^2+4n+1$$ is $$O(n^4)$$
* $$a_0+a_1n+\cdots+a_dn^d$$ is $$O(n^d)$$
* $$2n+100logn$$ is $$O(n)$$

```python
def find_max(data):
    # Return the maximum element from a nonempty Python list
    biggest = data[0] # The initial value to beat
    for val in data: # For each value:
        if val > biggest: # if it is greater than the best so far,
            biggest = val # we have found a new best (so far)
    return biggest # When loop ends, biggest is the max
    
```

* initialization: O(1)
* loop: O(n)
* return: O(1)

To sum up, this algorithm has O(n) time complexity.

```python
def prefix_average1(S):
    # Return list such tath, for all j, A[j] equals average of S[0], ..., S[j]
    n = len(S)
    A = [0]*n     # Create new list of n zeros
    for j in range(n):
        total = 0    # begin computing S[0]+...+S[j]
        for i in range(j+1):
            total += S[i]
        A[j] = total / (j+1)    # record the average
    return A
```

The running time of prefix\_average1 is $$O(n^2)$$

```python
def prefix_average2(S):
    n = len(S)
    A = [0]*n
    for j in range(n):
        A[j] = sum(S[0:j])/(j+1)
    return A
```

This big-Oh notation is used widely to characterize running times and space bounds in terms of some parameter n. (prefix\_average2 is also $$O(n^2)$$)

```python
def prefix_average3(S):
    n = len(S)
    A = [0] * n
    total = 0
    for j in range(n):
        total += S[j]
        A[j] = total / (j+1)
    return A
```

The above expression only has $$O(n)$$time complexity.



## Time complexity in Python

* len(data): O(1)
* data\[j]: O(1)

Python's lists are implemented as **array-based sequences.**



