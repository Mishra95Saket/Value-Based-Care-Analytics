-- readmission_logic.sql
WITH ordered AS (
  SELECT
    a.*,
    LEAD(admit_date) OVER (PARTITION BY member_id ORDER BY admit_date) AS next_admit_date,
    LEAD(admission_id) OVER (PARTITION BY member_id ORDER BY admit_date) AS next_admission_id
  FROM admissions a
),
flags AS (
  SELECT
    *,
    DATE_DIFF('day', discharge_date, next_admit_date) AS days_to_next_admit,
    CASE
      WHEN DATE_DIFF('day', discharge_date, next_admit_date) BETWEEN 1 AND 30 THEN 1 ELSE 0
    END AS is_30d_readmission
  FROM ordered
)
SELECT * FROM flags;
