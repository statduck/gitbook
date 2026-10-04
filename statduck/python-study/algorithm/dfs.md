# DFS

Depth First Search



```python

def dfs(graph, start_node):
    visit = list() # Emtpy list
    stack = list() # Emtpy list

    stack.append(start_node)

    while stack: # When one element of stack is inserted
        node = stack.pop() # The last element of stack
        if node not in visit: 
            visit.append(node)
            stack.extend(graph[node])

    return visit

graph = {'A': ['B'], 'B': ['A','B']}
print(dfs(graph, 'A'))
```







