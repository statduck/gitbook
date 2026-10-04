---
description: Object-Oriented Programming
---

# Class

In the object-oriented paradigm each object is an instance of a class.

## Class

2.3.1 Example: CreditCard Class

&#x20;**The Self Identifier**&#x20;

```python
class CreditCard:
    def __init__(self, customer, bank, acnt, limit, apr):
        self._customer = customer
        self._bank = bank
        self._account = acnt
        self._limit = limit
        self._balance = 0
        
    def get_customer(self):
        return self._customer
    
    def get_bank(self):
        return self._bank
        
    def get_account(self):
        return self._account
    
    def get_limit(self):
        return self._limit
        
    def get_balance(self):
        return self._balance
        
    def charge(self,price):
     if price + self._balance > self._limit:
         return False
     else:
         self._balance += price
         return True
         
     def make_payment(self, amount):
         self.baance -= amount
 
```

&#x20;**The Constructor**

A user can create an instance of the CreditCard class using a syntax as:

cc = Credit('John Doe', '1st bank')

변수명에 \_붙이는 이유: [https://mingrammer.com/underscore-in-python/](https://mingrammer.com/underscore-in-python/)

**Iterators**&#x20;

```python
class SequenceIterator:
    def __init__(self, sequence):
        self._seq = sequence
        self._k=-1
        
    def __next__(self):
    ## Return the next element, or else raise StopIteration error.
    self._k += 1
    if self._k < len(self.seq):
        return(self._seq[self._k])
    else:
        raise StopIteration()
    
    def __iter__(self):
        return self
```

```python
class PredatoryCreditCard(CreditCard):
    def __init__(self, customer, bank, acnt, limit, apr):
    
        super().__init__(customer,bank)
        self._apr = apr
    
    def charge(self, price):
        success = super().charger(price)
        if not success:
            self._balance += 5
        return success
        
    def process_month(self):
        if self._balance > 0:
            monthly_factor = pow(1+self._apr, 1/12)
            self._balance *= monthly_factor
```

super() is calling the inherited constructors. This method calls the CreditCard superclass.
