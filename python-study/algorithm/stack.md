# Stack

[Baekjoon 10799 - Iron rod](https://www.acmicpc.net/problem/10799)

```python
rod = list(input())
stack = []; sum_stack = 0; pre = ')'

for elem in rod:
    if elem == '(': stack.append('(')
    else: 
        if elem == pre: stack.pop(); sum_stack += 1 # ))
        else: 
            stack.pop(); sum_stack += len(stack)  # ()
    pre = elem

print(sum_stack)
```









