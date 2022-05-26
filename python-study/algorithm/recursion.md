# Recursion

## Definition

&#x20;Recursion is a technique by which a function makes one or more calls to itself during execution, or by which a data structure relies upon smaller instances of the very same type of structure in its representation.

## Example

&#x20;The representative examples are below

1. The **factorial function**
2. An **English ruler**
3. **Binary search**

### Factorial function



![](<../../.gitbook/assets/image (44).png>)

```python
def factorial(n):
    if n==0:
        return 1
    else:
        return n*factorial(n-1)
```

###

### Drawing an English Ruler

![](<../../.gitbook/assets/image (45).png>)

```python
def draw_line(tick_length, tick_label=''):
    # Draw one line with given tick length.
    line = '-' * tich_length
    if tick_label:
        line+=' '+tick_label
    print(line)
    
def draw_interval(center_length):
    # Draw tick interval based upon a central tick length.
    if center_length > 0: # stop when length drops to 0
        draw_interval(center_length -1) # recursively draw top ticks
        draw_line(center_length) # draw center tick
        draw_interval(center_length-1) # recursively draw bottom ticks

def draw_ruler(num_inches, major_length):
    # Draw English ruler
    draw_line(major_length, '0') # draw inch 0 line
    for j in range(1, 1+num_inches):
        draw_interval(major_length-1) # draw interior ticks for inch
        draw_line(major_length, str(j)) # draw inch j line and label

```

![](<../../.gitbook/assets/image (46).png>)



### Binary Search

&#x20;Binary Search is the method that used to efficiently locate a target value within a sorted sequence of n elements.(If it is not sorted, we use a sequential search algorithm.)

* If the target equals data\[mid], then we have found the item we are looking for, and the search terminates successfully.
* If target < data\[mid], then we recur on the first half of the sequence, that is, on the interval of indices from low to mid-1.
* If target > data\[mid], then we recur on the second half of the sequence, that is, on the interval of indices from mid+1 to high.

![](<../../.gitbook/assets/image (47).png>)

```python
def binary_search(data, target, low, high):
    # Return True if target is found in indicated portion of a Python list.
    # The search only considers the portions from data[low] to data[high] inclusive.
    if low>high:
        return False #interval is empty; no match
    else:
        mid=(low+high)//2
        if target==data[mid]: # found a match
            return True
        elif target < data[mid]:
            # recur on the portion left on the middle
            return binary_search(data, target, low, mid-1)
        else:
            # recur on the portion right of the middle
            return binary_search(data, target,mid+1,high) 
```

![](<../../.gitbook/assets/image (49).png>)



## Recursion Run Amok

Fibonacci Numbers

$$
F_0=0 \\
F_1=1 \\
F_n = F_{n-2}+F_{n-1} \; for \; n>1
$$

```python
def bad_fibonacci(n):
    # Return the nth Fibonacci number
    if n<=1:
        return n
    else:
        return bad_fibonacci(n-2)+bad_fibonacci(n-1)
```

$$
\begin{split}
c_0 = & 1 \\
c_1 = & 1 \\
c_2 = & 1+c_0+c_1=1+1+1=3 \\
c_3 = & 1+c_1+c_2=1+1+3=5 \\
c_4 = & 1+c_2+c_3=1+3+5=9

\end{split}
$$

exponential in n.

```python
def good_fibonacci(n)
    # Return pair of Fibonacci numbers, F(n) and F(n-1)
    if n<=1:
        return(n,0)
    else:
        (a,b)=good_fibonacci(n-1)
        return(a+b,a)
```

Linear recursion

0 1 1 2





&#x20;We don't have to use a recursive function twice. Instead of using twice, we just let the return value have pair structure. By doing this, the time complexity is reduced from exponential form to linear form.



## Exercises

Linear recursion can be a useful tool for processing a data sequence, such as a Python list.



**Q1. Let's compute the sum of a sequence S, of n integers.**

```python
def linear_sum(S,n):
    # Return the sum of the first n numbers of sequence S
    if n==0:
        return 0
    else:
        return linear_sum(S,n-1)+S[n-1]
```

S=\[4,3,6,2,8], linear\_sum(S,5)=23



**Q2. Write a short recursive Python function that finds the minimum and maximum values in a sequence without using any loops.**

****

**Q3. Give a recursive algorithm to compute the product of two positive integers, m and n, using only addition and subtraction.**

****

****

****

****



