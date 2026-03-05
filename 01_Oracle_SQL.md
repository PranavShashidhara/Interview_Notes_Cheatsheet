# Oracle SQL Cheatsheet

## Core SQL Fundamentals

### Data Types
```sql
NUMBER(p,s)       -- Numeric: precision p, scale s
VARCHAR2(n)       -- Variable-length string up to n bytes
DATE              -- Date + time (no timezone)
TIMESTAMP         -- Date + time with fractional seconds
CLOB              -- Character large object
BLOB              -- Binary large object
```

### DDL
```sql
CREATE TABLE employees (
    emp_id   NUMBER PRIMARY KEY,
    name     VARCHAR2(100) NOT NULL,
    dept_id  NUMBER REFERENCES departments(dept_id),
    salary   NUMBER(10,2),
    hire_dt  DATE DEFAULT SYSDATE
);

ALTER TABLE employees ADD (email VARCHAR2(200));
ALTER TABLE employees MODIFY (salary NUMBER(12,2));
ALTER TABLE employees DROP COLUMN email;

TRUNCATE TABLE employees;   -- Fast, no rollback
DROP TABLE employees;
```

### DML
```sql
INSERT INTO employees (emp_id, name, salary) VALUES (1, 'Alice', 90000);
UPDATE employees SET salary = salary * 1.1 WHERE dept_id = 10;
DELETE FROM employees WHERE hire_dt < DATE '2020-01-01';
MERGE INTO target t
  USING source s ON (t.id = s.id)
  WHEN MATCHED THEN UPDATE SET t.val = s.val
  WHEN NOT MATCHED THEN INSERT (id, val) VALUES (s.id, s.val);
```

---

## Joins

```sql
-- INNER JOIN
SELECT e.name, d.dept_name
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id;

-- LEFT OUTER JOIN
SELECT e.name, d.dept_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id;

-- FULL OUTER JOIN
SELECT e.name, d.dept_name
FROM employees e
FULL OUTER JOIN departments d ON e.dept_id = d.dept_id;

-- SELF JOIN
SELECT a.name manager, b.name employee
FROM employees a JOIN employees b ON b.manager_id = a.emp_id;

-- CROSS JOIN
SELECT e.name, d.dept_name FROM employees e CROSS JOIN departments d;
```

---

## Subqueries

```sql
-- Scalar subquery
SELECT name, (SELECT AVG(salary) FROM employees) avg_sal FROM employees;

-- Correlated subquery
SELECT name FROM employees e
WHERE salary > (SELECT AVG(salary) FROM employees WHERE dept_id = e.dept_id);

-- EXISTS
SELECT name FROM employees e
WHERE EXISTS (SELECT 1 FROM projects p WHERE p.emp_id = e.emp_id);

-- IN / NOT IN
SELECT name FROM employees WHERE dept_id IN (10, 20, 30);
```

---

## WITH Clause (Common Table Expressions)

The WITH clause (CTE) materializes a named result set reused in the main query. Improves readability and can improve performance via query block reuse.

```sql
WITH dept_avg AS (
    SELECT dept_id, AVG(salary) avg_sal
    FROM employees
    GROUP BY dept_id
),
high_earners AS (
    SELECT e.emp_id, e.name, e.salary, d.avg_sal
    FROM employees e
    JOIN dept_avg d ON e.dept_id = d.dept_id
    WHERE e.salary > d.avg_sal * 1.2
)
SELECT * FROM high_earners ORDER BY salary DESC;
```

### Recursive CTE (Hierarchical Data)
```sql
WITH RECURSIVE emp_hier (emp_id, name, manager_id, level) AS (
    -- Anchor: top-level employees
    SELECT emp_id, name, manager_id, 1
    FROM employees WHERE manager_id IS NULL
    UNION ALL
    -- Recursive: subordinates
    SELECT e.emp_id, e.name, e.manager_id, h.level + 1
    FROM employees e
    JOIN emp_hier h ON e.manager_id = h.emp_id
)
SELECT * FROM emp_hier ORDER BY level, name;

-- Oracle-specific hierarchical query (alternative)
SELECT LEVEL, emp_id, name, manager_id
FROM employees
START WITH manager_id IS NULL
CONNECT BY PRIOR emp_id = manager_id
ORDER SIBLINGS BY name;
```

---

## Window Functions (OVER / PARTITION BY)

Window functions compute results across a set of rows related to the current row without collapsing them.

### Syntax
```sql
function_name() OVER (
    [PARTITION BY col1, col2]
    [ORDER BY col3 ASC|DESC]
    [ROWS|RANGE BETWEEN frame_start AND frame_end]
)
```

### Ranking Functions
```sql
SELECT
    emp_id,
    name,
    salary,
    dept_id,
    RANK()         OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rank,
    DENSE_RANK()   OVER (PARTITION BY dept_id ORDER BY salary DESC) AS dense_rank,
    ROW_NUMBER()   OVER (PARTITION BY dept_id ORDER BY salary DESC) AS row_num,
    NTILE(4)       OVER (ORDER BY salary DESC)                      AS quartile
FROM employees;
-- RANK:        gaps after ties (1,1,3)
-- DENSE_RANK:  no gaps (1,1,2)
-- ROW_NUMBER:  unique sequential (1,2,3)
```

### Aggregate Window Functions
```sql
SELECT
    name,
    salary,
    dept_id,
    SUM(salary)    OVER (PARTITION BY dept_id)                          AS dept_total,
    AVG(salary)    OVER (PARTITION BY dept_id)                          AS dept_avg,
    COUNT(*)       OVER (PARTITION BY dept_id)                          AS dept_count,
    MAX(salary)    OVER (PARTITION BY dept_id)                          AS dept_max,
    SUM(salary)    OVER (ORDER BY hire_dt ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total
FROM employees;
```

### Lead / Lag (Access Adjacent Rows)
```sql
SELECT
    name,
    hire_dt,
    salary,
    LAG(salary, 1, 0)  OVER (PARTITION BY dept_id ORDER BY hire_dt) AS prev_salary,
    LEAD(salary, 1, 0) OVER (PARTITION BY dept_id ORDER BY hire_dt) AS next_salary,
    salary - LAG(salary) OVER (ORDER BY hire_dt)                    AS salary_change
FROM employees;
```

### First / Last Value
```sql
SELECT
    name,
    salary,
    FIRST_VALUE(name) OVER (PARTITION BY dept_id ORDER BY salary DESC) AS top_earner,
    LAST_VALUE(name)  OVER (
        PARTITION BY dept_id ORDER BY salary DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS lowest_earner
FROM employees;
```

### Frame Clauses
```sql
-- Rolling 3-row average
AVG(salary) OVER (ORDER BY hire_dt ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)

-- All rows in partition
SUM(salary) OVER (PARTITION BY dept_id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)

-- Range: rows with same ORDER BY value
SUM(salary) OVER (ORDER BY hire_dt RANGE BETWEEN INTERVAL '7' DAY PRECEDING AND CURRENT ROW)
```

---

## Analytical / ETL-Relevant Functions

```sql
-- LISTAGG: aggregate strings
SELECT dept_id, LISTAGG(name, ', ') WITHIN GROUP (ORDER BY name) AS members
FROM employees GROUP BY dept_id;

-- PIVOT
SELECT * FROM (
    SELECT dept_id, job_title, salary FROM employees
)
PIVOT (AVG(salary) FOR job_title IN ('Analyst' AS analyst, 'Manager' AS manager));

-- UNPIVOT
SELECT dept_id, metric, value
FROM dept_stats
UNPIVOT (value FOR metric IN (q1_sales AS 'Q1', q2_sales AS 'Q2'));

-- CASE expression
SELECT name,
       CASE WHEN salary > 100000 THEN 'High'
            WHEN salary > 60000  THEN 'Mid'
            ELSE 'Low' END AS band
FROM employees;

-- NULLIF, NVL, COALESCE
SELECT COALESCE(commission, bonus, 0) AS comp FROM employees;
SELECT NVL(commission, 0) FROM employees;
SELECT NULLIF(hours_worked, 0) FROM timesheets;  -- returns NULL if equal
```

---

## Query Optimization

### Execution Plan
```sql
EXPLAIN PLAN FOR
SELECT * FROM employees e JOIN departments d ON e.dept_id = d.dept_id;
SELECT * FROM TABLE(DBMS_XPLAN.DISPLAY);
```

### Index Types
```sql
CREATE INDEX idx_emp_dept ON employees(dept_id);
CREATE UNIQUE INDEX idx_emp_email ON employees(email);
CREATE BITMAP INDEX idx_emp_gender ON employees(gender);  -- low cardinality
CREATE INDEX idx_emp_name_upper ON employees(UPPER(name)); -- function-based
```

### Optimization Techniques
- Use bind variables to avoid hard parsing
- Avoid functions on indexed columns in WHERE: `WHERE salary + 0 = 90000` breaks index use; use `WHERE salary = 90000`
- Partition large tables (RANGE, LIST, HASH, COMPOSITE) to enable partition pruning
- Use `EXISTS` instead of `IN` for large subqueries (stops at first match)
- Prefer `UNION ALL` over `UNION` when duplicates are not a concern (no sort/dedup step)
- `GATHER_STATS_JOB` / `DBMS_STATS` keep optimizer statistics current

### Partitioning (Relevant to 200GB+ ETL work)
```sql
CREATE TABLE sales (
    sale_id  NUMBER,
    sale_dt  DATE,
    amount   NUMBER
)
PARTITION BY RANGE (sale_dt) (
    PARTITION p2022 VALUES LESS THAN (DATE '2023-01-01'),
    PARTITION p2023 VALUES LESS THAN (DATE '2024-01-01'),
    PARTITION p_max VALUES LESS THAN (MAXVALUE)
);
```

---

## PL/SQL Essentials

```sql
-- Stored procedure
CREATE OR REPLACE PROCEDURE update_salaries (p_dept_id IN NUMBER, p_pct IN NUMBER) IS
BEGIN
    UPDATE employees SET salary = salary * (1 + p_pct/100)
    WHERE dept_id = p_dept_id;
    COMMIT;
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        RAISE;
END;

-- Cursor
DECLARE
    CURSOR c_emp IS SELECT emp_id, name FROM employees WHERE dept_id = 10;
BEGIN
    FOR rec IN c_emp LOOP
        DBMS_OUTPUT.PUT_LINE(rec.name);
    END LOOP;
END;

-- Bulk collect for performance (used in large ETL)
DECLARE
    TYPE emp_tab IS TABLE OF employees%ROWTYPE;
    l_emps emp_tab;
BEGIN
    SELECT * BULK COLLECT INTO l_emps FROM employees WHERE dept_id = 10;
    FORALL i IN 1..l_emps.COUNT
        UPDATE archive_emp SET salary = l_emps(i).salary
        WHERE emp_id = l_emps(i).emp_id;
END;
```

---

## ETL-Specific Patterns (Resume Context)

```sql
-- Incremental load: only new/changed records
INSERT INTO target_table
SELECT * FROM source_table s
WHERE s.modified_dt > (SELECT MAX(load_dt) FROM target_table);

-- Deduplication using ROW_NUMBER
WITH deduped AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY business_key ORDER BY modified_dt DESC) rn
    FROM staging_table
)
INSERT INTO target SELECT * FROM deduped WHERE rn = 1;

-- Slowly Changing Dimension Type 2
MERGE INTO dim_employee t
USING staging s ON (t.emp_id = s.emp_id AND t.is_current = 'Y')
WHEN MATCHED AND (t.salary != s.salary) THEN
    UPDATE SET t.is_current = 'N', t.end_date = SYSDATE
WHEN NOT MATCHED THEN
    INSERT (emp_id, salary, start_date, end_date, is_current)
    VALUES (s.emp_id, s.salary, SYSDATE, NULL, 'Y');
```

---

## Key Interview Points

- **WITH vs subquery**: CTE improves readability; Oracle can materialize it (once) or inline it; use `/*+ MATERIALIZE */` or `/*+ INLINE */` hints to control
- **RANK vs DENSE_RANK vs ROW_NUMBER**: gaps vs no-gaps vs always-unique
- **PARTITION BY vs GROUP BY**: OVER/PARTITION BY does not collapse rows; GROUP BY does
- **Correlated subquery cost**: re-executes per row — replace with JOIN or window function when possible
- **ROWNUM vs ROW_NUMBER**: ROWNUM is assigned before ORDER BY; use ROW_NUMBER() OVER (ORDER BY ...) for reliable top-N
- **Transaction control**: COMMIT, ROLLBACK, SAVEPOINT; DDL auto-commits in Oracle
- **EXPLAIN PLAN**: always check for full table scans on large tables — add indexes or partition pruning
