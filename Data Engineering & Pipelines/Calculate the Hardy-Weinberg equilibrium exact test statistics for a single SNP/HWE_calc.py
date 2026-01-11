#Orginal citation: Wigginton, JE, Cutler, DJ, and Abecasis, GR (2005) A Note on Exact Tests of Hardy-Weinberg Equilibrium. AJHG 76: 887-893

def snphwe(obs_hets: int, obs_hom1: int, obs_hom2: int) -> float:
"""
Calculate Hardy-Weinberg equilibrium exact test for a variant

Args:
    obs_hets (int): observed heterozygous number
    obs_hom1 (int): first observed homozygous number
    obs_hom2 (int): second observed homozygous number

Returns:
    float: Hardy-Weinberg equilibrium exact test p-value.

    - P > 0.05: No significant deviation from Hardy-Weinberg equilibrium (HWE).
    - P ≤ 0.05: Significant deviation from HWE, which could be due to factors such as population stratification, inbreeding, genotyping errors, or natural selection.
"""

# Calculate the number of samples
obs_minor_homs = min(obs_hom1, obs_hom2)
obs_mijor_homs = max(obs_hom1, obs_hom2)

# Calculate the number of rare alleles
rare = 2 * obs_minor_homs + obs_hets
# Calculate the number of genotypes
genotypes = obs_hets + obs_mijor_homs + obs_minor_homs

# Initialize the probability of each possible genotype
probs = [0.0 for i in range(rare +1)]

# Calculate the mid value
mid = rare * (2 * genotypes - rare) // (2 * genotypes)

# If the number of rare alleles is odd, adjust the mid value
if mid % 2 != rare%2:
    mid += 1
# Initialize the probability of the mid value
probs[mid] = 1.0
sum_p = 1 #probs[mid]


curr_homr = (rare - mid) // 2
curr_homc = genotypes - mid - curr_homr

# 
for curr_hets in range(mid, 1, -2):
    probs[curr_hets - 2] = probs[curr_hets] * curr_hets * (curr_hets - 1.0)/ (4.0 * (curr_homr + 1.0) * (curr_homc + 1.0))
    sum_p+= probs[curr_hets - 2]
    curr_homr += 1
    curr_homc += 1

curr_homr = (rare - mid) // 2
curr_homc = genotypes - mid - curr_homr

# 
for curr_hets in range(mid, rare-1, 2):
    probs[curr_hets + 2] = probs[curr_hets] * 4.0 * curr_homr * curr_homc/ ((curr_hets + 2.0) * (curr_hets + 1.0))
    sum_p += probs[curr_hets + 2]
    curr_homr -= 1
    curr_homc -= 1

target = probs[obs_hets]
p_hwe = 0.0
for p in probs:
    if p <= target :
        p_hwe += p / sum_p  

return min(p_hwe,1)

if __name__ == '__main__':
# For an example, we had 502 samples. 
# At position 1:14930:A:G, we count:
#     + 407 heterozygous genotypes (signed 0|1 or 1|0).
#     + 4 homozygous genotypes AA (signed 0|0)
#     + 91 homozygous genotypes GG (signed 1|1)
print(snphwe(407,4,91))