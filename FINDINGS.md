# FINDINGS

## Dataset Description
- Final dataset shape: (2000, 10)
- Missing values in `commute_minutes` were filled using the median due to the presence of missing values.
- No missing values were observed in `study_hours_weekly` in this dataset.
- Missing values were also present in `scholarship`.
- See `output/data_profile.txt` for full details.

## Distribution Findings
- GPA distribution was explored using a histogram and appears concentrated around mid-range values.
- Study hours and attendance were also plotted to understand their spread.
- GPA differences across departments were visualized using a box plot, showing variation between departments.
- Scholarship categories were visualized using a bar chart.

## Correlation Findings
- `study_hours_weekly` and `gpa`: correlation = 0.639 → moderate positive relationship (students who study more tend to have higher GPA).
- `gpa` and `attendance_pct`: correlation = 0.041 → very weak relationship.
- Correlation does not imply causation.

## Hypothesis Test Results

### Internship and GPA
- Mean GPA (Yes): 2.983
- Mean GPA (No): 2.701
- t-statistic: 14.229
- p-value: 0.000000
- Cohen's d: 0.690

**Interpretation:**  
The result is statistically significant (p < 0.05). Students with internships have higher GPA on average.  
The effect size (Cohen’s d = 0.690) indicates a moderate to strong practical impact.

---

### Scholarship and Department
- Chi-square statistic: 13.949
- p-value: 0.304005
- Degrees of freedom: 12

**Interpretation:**  
The result is not statistically significant (p > 0.05), meaning there is no strong evidence of an association between scholarship type and department.

---

## Recommendations
1. Increase internship opportunities for students, as internships are associated with higher GPA.
2. Provide additional academic support for students with lower GPA.
3. Encourage effective study habits, since study hours show a moderate positive relationship with GPA.