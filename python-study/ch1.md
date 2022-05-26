# Python OverView

## 1.1 Python Overview

Python is formally an interpreted language. Commands are executed through the Python interpreter. Programmer usually uses the source code or script to save his command.

```python
print( Welcome to the GPA calculator. )
print( Please enter all your letter grades, one per line. ) print( Enter a blank line to designate the end. )
# map from letter grade to point value
points = { A+ :4.0, A :4.0, A- :3.67, B+ :3.33, B :3.0, B- :2.67,
C+ :2.33, C :2.0, C :1.67, D+ :1.33, D :1.0, F :0.0}
num courses = 0
total points = 0
done = False
while not done:
   grade = input( )   # read line from user
   if grade == :   # empty line was entered
      done = True
   elif grade not in points:   # unrecognized grade entered
      print("Unknown grade {0} being ignored".format(grade))
      else:
   num courses += 1
total points += points[grade]
if num courses > 0: # avoid division by zero
print( Your GPA is {0:.3} .format(total points / num courses))
```

## 1.2 Objects in Python

**Assignment statement**

```python
temperature = 98.6
```

It makes this temperature as an **identifier** and then associates it with the object expressed on the right-hand side of the equal sine.

Because Python is a **dynamically typed language** it associates an identifier with any type of object.

### Instantiation

The process of creating a new instance of a class is known as **instantiation.** If there were a class named Widget, we could create an instance of that class using a syntax such as w=Widget()



