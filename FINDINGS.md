# FINDINGS

## Dataset Description
- Final dataset shape: (2000, 10)
- Missing values in commute_minutes were filled with the median.
- Missing values in study_hours_weekly were dropped.
- See output/data_profile.txt for full details.

## Distribution Findings
- GPA distribution was explored using a histogram.
- Study hours and attendance were also plotted.
- GPA differences across departments were shown using a box plot.
- Scholarship categories were shown using a bar chart.

## Correlation Findings
- study_hours_weekly and gpa: correlation = 0.639
- gpa and attendance_pct: correlation = 0.041
- Correlation does not imply causation.

## Hypothesis Test Results
### Internship and GPA
- Mean GPA (Yes): 2.983
- Mean GPA (No): 2.701
- t-statistic: 14.229
- p-value: 0.000000
- Cohen's d: 0.690

### Scholarship and Department
- Chi-square statistic: 13.949
- p-value: 0.304005
- Degrees of freedom: 12

## Recommendations
1. Increase internship opportunities for students.
2. Support students with lower GPA through tutoring or advising.
3. Encourage better study habits because study hours may be linked to GPA.
