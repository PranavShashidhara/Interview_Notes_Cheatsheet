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

## Statistical SQL Functions

```sql
-- Median
SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS median_salary
FROM employees;

-- Standard deviation and variance
SELECT department_id,
       STDDEV_POP(salary) AS pop_stddev,
       STDDEV_SAMP(salary) AS sample_stddev,
       VARIANCE(salary) AS variance
FROM employees GROUP BY department_id;

-- Correlation and linear regression in SQL
SELECT CORR(years_experience, salary) AS correlation,
       REGR_SLOPE(salary, years_experience) AS slope,
       REGR_INTERCEPT(salary, years_experience) AS intercept,
       REGR_R2(salary, years_experience) AS r_squared
FROM employees;

-- Percentiles (IQR for outlier detection)
SELECT department_id,
       PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY salary) AS p25,
       PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY salary) AS p75
FROM employees GROUP BY department_id;
```

---

## NULL Traps — Classic Interview Gotchas

- **NOT IN with NULLs**: `WHERE id NOT IN (SELECT id FROM t)` returns **NO rows** if subquery contains any NULL. Use `NOT EXISTS` instead.
- **COUNT(*) vs COUNT(col)**: `COUNT(*)` counts all rows; `COUNT(col)` skips NULLs. This changes results silently.
- **Aggregations ignore NULLs**: `AVG(col)` ignores NULLs, so `AVG(10, NULL, 20)` = 15, not 10.
- **NULL in CASE WHEN**: `CASE WHEN col = 'X'` does NOT match NULL. Use `CASE WHEN col IS NULL`.
- **Three-valued logic**: `NULL = NULL` is NULL (not TRUE). Use `IS NULL` / `IS NOT NULL`.
- **COALESCE vs NVL**: `COALESCE` is ANSI standard, short-circuits; `NVL` is Oracle-specific, evaluates both args.
- **ORDER BY with NULLs**: NULLs sort last by default in Oracle (first in PostgreSQL). Use `NULLS FIRST` / `NULLS LAST`.
- **JOIN with NULLs**: `NULL = NULL` in JOIN condition never matches. Use `NVL` or `COALESCE` to handle.

---

## Time-Series SQL Patterns

### Sessionization (Identifying User Sessions)
```sql
-- Mark new session when gap > 30 minutes
WITH activity AS (
  SELECT user_id, event_time,
         LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time) AS prev_time
  FROM events
),
sessions AS (
  SELECT user_id, event_time,
         SUM(CASE WHEN event_time - prev_time > INTERVAL '30' MINUTE
                   OR prev_time IS NULL THEN 1 ELSE 0 END)
           OVER (PARTITION BY user_id ORDER BY event_time) AS session_id
  FROM activity
)
SELECT user_id, session_id,
       MIN(event_time) AS session_start,
       MAX(event_time) AS session_end,
       COUNT(*) AS events
FROM sessions GROUP BY user_id, session_id;
```

### Consecutive Days (Gap-and-Island)
```sql
-- Find users active 3+ consecutive days
WITH grouped AS (
  SELECT user_id, activity_date,
         activity_date - ROW_NUMBER() OVER (
           PARTITION BY user_id ORDER BY activity_date
         ) * INTERVAL '1' DAY AS grp
  FROM daily_active_users
)
SELECT user_id,
       MIN(activity_date) AS streak_start,
       MAX(activity_date) AS streak_end,
       COUNT(*) AS streak_length
FROM grouped
GROUP BY user_id, grp
HAVING COUNT(*) >= 3;
```

### Rolling Aggregates
```sql
-- 7-day rolling average revenue
SELECT order_date,
       AVG(daily_revenue) OVER (
         ORDER BY order_date
         ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
       ) AS rolling_7d_avg
FROM daily_revenue;
```

---

## Funnel and Cohort Analysis

### Conversion Funnel
```sql
SELECT
  COUNT(DISTINCT CASE WHEN event = 'view' THEN user_id END) AS viewed,
  COUNT(DISTINCT CASE WHEN event = 'add_to_cart' THEN user_id END) AS carted,
  COUNT(DISTINCT CASE WHEN event = 'purchase' THEN user_id END) AS purchased,
  ROUND(COUNT(DISTINCT CASE WHEN event = 'add_to_cart' THEN user_id END) * 100.0 /
        NULLIF(COUNT(DISTINCT CASE WHEN event = 'view' THEN user_id END), 0), 1) AS view_to_cart_pct,
  ROUND(COUNT(DISTINCT CASE WHEN event = 'purchase' THEN user_id END) * 100.0 /
        NULLIF(COUNT(DISTINCT CASE WHEN event = 'add_to_cart' THEN user_id END), 0), 1) AS cart_to_purchase_pct
FROM events
WHERE event_date BETWEEN DATE '2024-01-01' AND DATE '2024-01-31';
```

### Retention Cohort Analysis
```sql
WITH cohorts AS (
  SELECT user_id, TRUNC(signup_date, 'MM') AS cohort_month
  FROM users
),
activity AS (
  SELECT DISTINCT user_id, TRUNC(activity_date, 'MM') AS activity_month
  FROM user_activity
)
SELECT c.cohort_month,
       MONTHS_BETWEEN(a.activity_month, c.cohort_month) AS month_number,
       COUNT(DISTINCT a.user_id) AS retained_users
FROM cohorts c
JOIN activity a ON c.user_id = a.user_id
  AND a.activity_month >= c.cohort_month
GROUP BY c.cohort_month, MONTHS_BETWEEN(a.activity_month, c.cohort_month)
ORDER BY c.cohort_month, month_number;
```

### DAU / MAU Stickiness Ratio
```sql
SELECT activity_date,
       COUNT(DISTINCT user_id) AS dau,
       COUNT(DISTINCT user_id) OVER (
         ORDER BY activity_date
         RANGE BETWEEN INTERVAL '29' DAY PRECEDING AND CURRENT ROW
       ) AS mau_approx
FROM user_activity
GROUP BY activity_date;
```

---

## SQL ↔ Pandas Quick Reference

| SQL | Pandas |
|---|---|
| `SELECT col1, col2 FROM t` | `df[['col1', 'col2']]` |
| `WHERE col > 5` | `df[df['col'] > 5]` |
| `GROUP BY col` | `df.groupby('col')` |
| `ORDER BY col DESC` | `df.sort_values('col', ascending=False)` |
| `JOIN t2 ON t1.id = t2.id` | `df1.merge(df2, on='id')` |
| `LEFT JOIN` | `df1.merge(df2, on='id', how='left')` |
| `COUNT(*), SUM(col) GROUP BY g` | `df.groupby('g').agg({'col': ['count', 'sum']})` |
| `ROW_NUMBER() OVER (PARTITION BY g ORDER BY col)` | `df.groupby('g')['col'].rank(method='first')` |
| `LAG(col) OVER (ORDER BY date)` | `df['col'].shift(1)` |
| `HAVING COUNT(*) > 5` | `df.groupby('g').filter(lambda x: len(x) > 5)` |
| `CASE WHEN ... THEN ... END` | `np.where(cond, val1, val2)` |
| `PIVOT` | `df.pivot_table(values, index, columns)` |

---

## Dialect Portability Notes

| Feature | Oracle | ANSI/PostgreSQL | BigQuery/Snowflake |
|---|---|---|---|
| Row limiting | `ROWNUM <= N` or `FETCH FIRST N ROWS ONLY` (12c+) | `LIMIT N` | `LIMIT N` |
| NULL handling | `NVL(a, b)` | `COALESCE(a, b)` | `IFNULL(a, b)` / `COALESCE` |
| String concat | `\|\|` or `CONCAT` | `\|\|` | `CONCAT` |
| Hierarchical | `CONNECT BY` / `START WITH` | Recursive CTE | Recursive CTE |
| Date diff | `date1 - date2` (returns days) | `DATE_PART('day', date1 - date2)` | `DATE_DIFF(d1, d2, DAY)` |
| Array unnest | N/A (use `TABLE()`) | `UNNEST(array_col)` | `UNNEST(array_col)` |
| Approx distinct | `APPROX_COUNT_DISTINCT` (12.1.0.2+) | N/A | `APPROX_COUNT_DISTINCT` |

---

## Interview Walkthrough Problems

### Problem 1: Second Highest Salary per Department
```sql
WITH ranked AS (
  SELECT department_id, employee_name, salary,
         DENSE_RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS rnk
  FROM employees
)
SELECT department_id, employee_name, salary
FROM ranked WHERE rnk = 2;
```
*Why DENSE_RANK over ROW_NUMBER?* — handles ties. If two employees share the highest salary, the second-highest is the next distinct value.

### Problem 2: Users Active 3 Consecutive Days
See the Gap-and-Island pattern above — this is one of the most frequently asked SQL problems at Meta, Airbnb, and Lyft.

### Problem 3: 7-Day Rolling Retention
```sql
SELECT TRUNC(u.signup_date, 'IW') AS signup_week,
       COUNT(DISTINCT u.user_id) AS signups,
       COUNT(DISTINCT a.user_id) AS retained,
       ROUND(COUNT(DISTINCT a.user_id) * 100.0 /
             NULLIF(COUNT(DISTINCT u.user_id), 0), 1) AS retention_pct
FROM users u
LEFT JOIN user_activity a ON u.user_id = a.user_id
  AND a.activity_date BETWEEN u.signup_date + 1 AND u.signup_date + 7
GROUP BY TRUNC(u.signup_date, 'IW');
```

### Problem 4: Running Total with Monthly Reset
```sql
SELECT order_date, revenue,
       SUM(revenue) OVER (
         PARTITION BY TRUNC(order_date, 'MM')
         ORDER BY order_date
       ) AS mtd_revenue
FROM orders;
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
