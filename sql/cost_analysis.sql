-- cost_analysis.sql
WITH ordered AS (
  SELECT
    a.*,
    LEAD(admit_date) OVER (PARTITION BY member_id ORDER BY admit_date) AS next_admit_date,
    LEAD(admission_id) OVER (PARTITION BY member_id ORDER BY admit_date) AS next_admission_id
  FROM admissions a
),
events AS (
  SELECT
    member_id,
    admission_id AS index_admission_id,
    primary_condition_group AS index_condition_group,
    preventable_proxy,
    discharge_date,
    next_admission_id,
    next_admit_date,
    DATE_DIFF('day', discharge_date, next_admit_date) AS days_to_readmit
  FROM ordered
)
SELECT
  index_condition_group,
  SUM(CASE WHEN preventable_proxy = 1 AND days_to_readmit BETWEEN 1 AND 30 THEN 1 ELSE 0 END) AS preventable_readmission_events
FROM events
GROUP BY index_condition_group
ORDER BY preventable_readmission_events DESC;
