# K-medoids\[PAM algorithm]

K-medoids method needs to compare the distance between two points in the data set, so time complexity is so enormous. To avoid the time complexity issue, we can use **PAM** algorithm.

&#x20;PAM: Partition Around Medoids.

\[ Notation ]

* $$O$$: The set of whole objects.
* $$S$$ : The set of selected objects.
* $$U$$ : The set of unselected objects
* $$D_p$$: The dissimilarity between $$p$$and the closest object in $$S$$
* $$E_p$$: The dissimilarity between $$p$$and the second closet object in $$S$$

U = O - S

\[ Process ]

**BUILD PHASE**

1. \[Initialization] Put the point for which the sum of the distances to all other objects is minimal into set S&#x20;
2. For$$i \in U$$, consider it as a candidate for $$S$$
3. For $$j\in U-\{i\}$$, compute $$D_j$$
4. If $$D_j > d(i, j)$$select object $$i$$, let $$C_{ji}=max\{D_j-d(j,i),0\}$$
5. Compute the total gain. $$g_i = \sum_{j\in U} C_{j}$$
6. Choose that object $$i$$that maximizes $$g_i$$
7. Let $$S := S \cup \{i\}$$and $$U = U-\{i\}$$
8. Repeat 1-7 until $$k$$objects have been selected.

```python
 def dist_others(i,j,target=['idx','dist','second_min']):
        # The input data i,j must be 2 dimension arrays
    i = np.array(i); j = np.array(j)
    if i.shape == (len(i),): # If it is 1 dimension
        i = np.expand_dims(i, axis=0)

    if j.shape == (len(j),): # If it is 1 dimension
        j = np.expand_dims(j, axis=0)

    if (len(i)>1)&(len(j)>1): # Multi to Multi
        ls = []; sum1=0
        for i_idx, elem1 in enumerate(i):
            for elem2 in j:
                diff = (elem1-elem2)**2
                sum1 += sum(diff)
            ls.append([i_idx,sum1])
        ls = np.array(ls)
        dist = np.min(ls, axis=0)[1]
        idx = np.argmin(ls, axis=0)[0]
        trg = {target=='idx':idx, target=='dist':dist}.get(True)
        return(trg)

    elif (len(i)==1)&(len(j)>1): # One to Multi
        diff = i-j
        sum1 = np.sum(diff**2, axis=1)
        idx = np.argmin(sum1); dist = np.min(sum1)
        try: second_min = sorted(set(sum1))[1]
        except : second_min = np.min(sum1)
        trg = {target=='idx':idx, target=='dist':dist, target=='second_min':second_min}.get(True)
        return(trg)

    elif (len(i)==1)&(len(j)==1): # One to One
        diff = i-j
        sum0 = np.sum(diff**2)
        dist = np.min(sum0)
        trg = {target=='idx':'No index in one to one case', target=='dist':dist}.get(True)
        return(trg)
        
# step2
def build(obj, sel):
    gi = 0; sel_i = 0; gi_sum=0; n = len(obj)
    for i in range(n):
        gpre = gi
        for j in range(n):
            if j==i: 
                continue
            gi = 0
            Dj = dist_others(obj[j],sel,'dist')
            dji = dist_others(obj[j],obj[i],'dist')
            Cji = max(Dj-dji,0)
            gi += Cji

        gi_sum += gi
        if (gpre<gi):
            sel_i = i

    obj = np.delete(obj, sel_i, axis=0)
    if (gi_sum==0): return('Stop')
    else: return(sel_i)

```

**SWAP PHASE**

This phase is the step switching the element in $$S$$ to one in $$U$$(The other way around also). It improves the quality of the clustering.&#x20;

Considers all pairs $$(i,h) \in S \times U$$

If $$d(j,i) > D_j$$

1. if $$d(j,h) \geq D_j$$, then $$K_{jih}=0$$
2. if $$d(j,h) < D_j$$, then $$K_{jih}=d(j,h)-D_j$$

In both subcases, $$K_{jih} = min\{d(j,h)-D_j,0\}$$

If $$d(j,i)=D_j$$

1. if $$d(j,h) < E_j$$, then $$K_{jih} = d(j,h)-D_j$$($$K$$ can be negative)
2. if $$d(j,h) \geq E_j$$, then $$K_{jih} = E_j-D_j$$ ( $$K > 0$$)

In both subcases, $$K_{jih} = min\{d(j,h),E_j\}-D_j$$

1. Compute the total result of swap as $$T_{ih}=\sum\{K_{ijh} | j\in U\}$$
2. Select a pair $$(i,h) \in S \times U$$that minimizes $$T_{ih}$$
3. If $$T_{ih}<0$$, the swap is carried out and $$D_p$$and$$E_p$$are updated for every object $$p$$. After it, return to Step1. If $$minT_{ih}>0$$,  the algorithm halts. (All $$T_{ih}>0$$ is also the halting condition.)

```python
def swap(obj,sel,i,h):
    n = len(obj); Tih = 0; Kjih = 0
    for j in range(n):
        dji = dist_others(obj[j],obj[i],'dist')
        Dj = dist_others(obj[j],sel,'dist')
        djh = dist_others(obj[j],obj[h],'dist')
        if dji>Dj:
            Kjih = min(djh-Dj,0)
        elif dji==Dj:
            Ej = dist_others(obj[j],sel,'second_min')
            Kjih = min(djh,Ej)-Dj
        Tih += Kjih
    return(Tih)
```

Time complexity of numpy expend\_dims is much bigger than using for statement.



ref: [https://www.cs.umb.edu/cs738/pam1.pdf](https://www.cs.umb.edu/cs738/pam1.pdf)
