---
description: list up useful statements, just find the statement you need with ctrl+f
---

# Basic Statement

Reference: [https://www.w3schools.com/sql/default.asp](https://www.w3schools.com/sql/default.asp)

This statement is for MySQL Syntax.



Select sub columns from the table

```sql
SELECT col1, col2
FROM table_name;
```



Select all columns from the table (\* means all)

```sql
SELECT * FROM table_name;
```



Select sub columns who have distanct values

```sql
SELECT DISTINCT col1, col2
FROM table_name;
```



Filter records based on a condition

```sql
SELECT col1, col2
FROM table_name
WHERE condition;
```

Operator:  =, >, <, >=, <=, <>, BETWEEN, LIKE(To find a pattern), IN(Multple values)

Connect several conditions with logical operator: AND, OR, NOT



Sort the table based on the col

```sql
SELECT col1, col2
FROM table_name
ORDER BY col1 ASC, col2 DESC; 
```



Insert new records into a table

```sql
INSERT INTO table_name (col1, col2)
VALUES (val1, val2);
```



Check for NULL

```sql
SELECT col1
FROM table_name
WHERE col1 IS NULL; # IS NOT NULL also can be used.
```



Update for modifying the existing records

```sql
UPDATE table_name
SET col1=val1, col2=val2
WHERE condition;
```

Under this condition, we modify the value from col1 with new value, so basically update statment is for changing the value in column.



Delete some records(row records)

```sql
DELETE FROM table_name WHERE condition;
```



Select top n data

```sql
SELECT col1
FROM table_name
WHERE condition
LIMIT number;
```

You can use more function like min, max, count, avg, and sum



Find the data which of column matches specific string pattern with regex.

```sql
SELECT col1, col2
FROM table_name
WHERE col1 LIKE pattern;
```

```sql
SELECT * FROM Customers
WHERE CustomerName LIKE 'a__%';
```



Aliases for giving name to table or column

(Because this statement is just temporary, so an alias only exists for the duration of that query.)

```sql
SELECT col1 AS alias_name 
FROM table_name;
```



