#####Chi-square & Fisher’s Exact Test: Recurrence vs. Levothyroxine Use#####
from scipy.stats import chi2_contingency, fisher_exact

# Contingency table
table = [[288, 1213],
         [9, 135]]

# Use chi-square
chi2, p, dof, expected = chi2_contingency(table)
print(f"Chi2: {chi2:.2f}, p-value: {p:.4f}")

# Use Fisher's exact test if small counts
odds_ratio, p_fisher = fisher_exact(table)
print(f"Fisher's Exact Test p-value: {p_fisher:.4f}, Odds Ratio: {odds_ratio:.2f}")

##———RESULTS———##
#Chi2: 14.00, p-value: 0.0002
#Fisher's Exact Test p-value: 0.0000, Odds Ratio: 3.56

#Statistically significant association between levothyroxine use and recurrence
#Patients taking Levothyroxine had 3.56 times higher odds of recurrence than those not taking it 


#####Welch’s t-test: Age Difference Between Levothyroxine Groups#####
import numpy as np
from scipy.stats import t

# Group 1: Levothyroxine
mean1 = 48.09
std1 = 15.33
n1 = 1501

# Group 2: No Levothyroxine
mean2 = 50.92
std2 = 14.85
n2 = 144

# Calculate t-statistic
se = np.sqrt((std1**2)/n1 + (std2**2)/n2)
t_stat = (mean1 - mean2) / se

# Degrees of freedom (Welch–Satterthwaite equation)
df = ((std1**2/n1 + std2**2/n2)**2) / (((std1**2/n1)**2)/(n1-1) + ((std2**2/n2)**2)/(n2-1))

# Two-tailed p-value
p_val = 2 * t.sf(np.abs(t_stat), df)

print(f"Welch’s t-test:\n  t = {t_stat:.4f}\n  df = {df:.2f}\n  p-value = {p_val:.4g}")

##———RESULTS———##
#Welch’s t-test:
#t = -2.1782
#df = 173.56
#p-value = 0.03074

#Statistically significant difference in average age: Levothyroxine users were younger on average


#####Chi-square: Sex vs. Recurrence#####
from scipy.stats import chi2_contingency
# Recurrence
# Female: 208 / 297
# Male: 88 / 297

# Total: Female: 1158, Male: 486

contingency = [[208, 1158-208],
               [88, 486-88]]

chi2, p, _, _ = chi2_contingency(contingency)
print(f"Sex vs. Recurrence: p = {p:.4f}")

##———RESULTS———##
#Sex vs. Recurrence: p = 1.0000

#No statistically significant association between sex and recurrence
