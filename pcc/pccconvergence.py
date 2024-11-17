import numpy as np
from scipy.stats import pearsonr, spearmanr
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import linregress

#IID
SV_1 = np.array([-24.409884888927145, -22.62767086327076, -23.909745270013808, -22.760746810833616, -23.29781094590823]) 
BV_1  = np.array([-5.948816725984216, -5.545571552589536, -5.311726374551654, -5.041963508352637, -5.593782817944884])
ABV_Simple_1 = np.array([-0.252875097328797, -0.2288856585510075, -0.2835275926627219, -0.2190842186100781, -0.2911171605810523])
ABV_Hessian_1 = np.array([-2.5707323076203465, -5.749352287501097, -22.565373002551496, 17.26112177222967, 18.256525065749884])

SV_2 = np.array([-23.173703097303715, -25.61744375824929, -24.713261716564496, -22.644227696458497, -24.378510726491612])
BV_2 = np.array([-4.706250285729766, -6.083594957366586, -5.773475898429751, -5.325923508033156, -6.170815190300345])
ABV_Simple_2 = np.array([-0.3507651600521058, -0.32443248899653554, -0.2853101100772619, -0.312060855794698, -0.3725827857851982])
ABV_Hessian_2 = np.array([-6.704698405228555, -9.542349236086011, 8.566787237301469, 13.532176872715354, 3.014599473681301])

SV_3 = np.array([-23.589924924572312, -22.529917253057164, -23.56882031559945, -22.05639503697554, -24.176807456215222])
BV_3 = np.array([-5.675594829022884, -5.37492698431015, -5.0323584489524364, -5.260698042809963, -5.779234848916531])
ABV_Simple_3 = np.array([-0.27311788592487574, -0.30650457204319537, -0.2528196880593896, -0.2694842256605625, -0.2510866450611502])
ABV_Hessian_3 = np.array([-12.284073442220688, 14.630857042968273, 10.21740753389895, -0.9845901178196073, -10.385668000206351])



# Aggregate across runs
SV_all = np.concatenate([SV_1, SV_2, SV_3])
ABV_Simple_all = np.concatenate([ABV_Simple_1, ABV_Simple_2, ABV_Simple_3])
ABV_Hessian_all = np.concatenate([ABV_Hessian_1, ABV_Hessian_2, ABV_Hessian_3])


from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Compute PCC
pcc_simple = pearsonr(SV_all, ABV_Simple_all)[0]
pcc_hessian = pearsonr(SV_all, ABV_Hessian_all)[0]

# Visualize relationships
plt.scatter(SV_all, ABV_Simple_all, label=f'PCC: {pcc_simple:.2f}')
plt.title('SV vs. ABV_Simple')
plt.xlabel('SV')
plt.ylabel('ABV_Simple')
plt.legend()
plt.show()


coefficients = np.polyfit(SV_all, ABV_Hessian_all, deg=2)  # Quadratic fit
poly_fit = np.poly1d(coefficients)


ABV_Hessian_log = np.log(np.abs(ABV_Hessian_all) + 1e-5)  # Avoid log(0)

plt.scatter(SV_all, ABV_Hessian_log)
plt.title('SV vs. log(ABV_Hessian)')
plt.xlabel('SV')
plt.ylabel('log(ABV_Hessian)')
plt.show()

ABV_Hessian_sqrt = np.sqrt(np.abs(ABV_Hessian_all)) * np.sign(ABV_Hessian_all)  # Preserve sign

plt.scatter(SV_all, ABV_Hessian_sqrt)
plt.title('SV vs. sqrt(ABV_Hessian)')
plt.xlabel('SV')
plt.ylabel('sqrt(ABV_Hessian)')
plt.show()


ABV_Hessian_exp = np.exp(ABV_Hessian_all)

plt.scatter(np.exp(SV_all), ABV_Hessian_exp)
plt.title('SV vs. exp(ABV_Hessian)')
plt.xlabel('SV')
plt.ylabel('exp(ABV_Hessian)')
plt.show()

slope, intercept, r_value, p_value, std_err = linregress(SV_all, ABV_Hessian_exp)

# Plot regression line
plt.scatter(SV_all, ABV_Hessian_all, label=f'PCC: {pcc_hessian:.2f}')
plt.plot(SV_all, slope * SV_all + intercept, color='red', label='Linear Fit')
plt.title('SV vs. ABV_Hessian')
plt.xlabel('SV')
plt.ylabel('ABV_Hessian')
plt.legend()
plt.show()

# NON IID:
# SV_1 = np.array([181.95880798796816, -38.19942807853222, -75.77778747876486, -35.91375678976377, -64.00016624927521])
# BV_1 = np.array([78.77293125167489, -21.987386893481016, -47.43576893955469, -27.594971112906933, -37.032924480736256])
# ABV_Simple_1 = np.array([1.141512457281351, -0.5619645267724991, -0.9291043318808079, -0.9759988049045205, -0.0395272308960557])
# ABV_Hessian_1 = np.array([643.036488163285, -80.68008410930634, -183.78021064121276, -141.8907394418493, -1.2421450885012746])

# SV_2 = np.array([-47.21579605738322, 156.01760527094203, -41.66360622048378, -65.57308938503266, -55.69278532663981])
# BV_2 = np.array([-26.833819838240743, 66.41561008058488, -19.051439123228192, -38.669827895238996, -28.168856808915734])
# ABV_Simple_2 = np.array([-0.9342706445604563, 1.087217248044908, -0.28548065200448036, -0.3898796336725354, -0.7880192287266254])
# ABV_Hessian_2 = np.array([-240.99354546517134, 573.586022160016, -39.10542896948755, -25.002182245254517, -98.89350616931915])

# SV_3 = np.array([-45.93091857532661, -74.23531784216563, 145.5069699873527, -75.99037101964154, -20.155097257097562])
# BV_3 = np.array([-20.130715768784285, -41.4580530077219, 68.27663829550147, -39.45971083268523, -9.218875374644995])
# ABV_Simple_3 = np.array([-0.8013164550065994, -0.7688949685543776, 1.17510259244591, -1.0703009562566876, -1.4710255786776543])
# ABV_Hessian_3 = np.array([-138.41716212034225, -75.8597601056099, 1094.956063770689, -187.7881381334737, -502.75566007755697])


# # MISLABELED:
# SV_1 = np.array([-20.208389015992488, -20.43079329232376, -21.672366650899257, -20.678431289394695, -22.19175741275151])
# BV_1 = np.array([-3.786082776263356, -3.7034053560346365, -4.443471597507596, -3.73646922968328, -4.792870158329606])
# ABV_Simple_1 = np.array([-0.30673281312920153, -0.33458197955042124, -0.3267051917500794, -0.26949214981868863, -0.3212974441703409])
# ABV_Hessian_1 = np.array([-1.550166587345302, -27.77011839300394, 12.43648948147893, -8.704493951052427, 6.646915657445788])

# SV_2 = np.array([-21.32336734036605, -22.076319419344266, -20.755728660027184, -21.11509440143903, -16.638341742753983])
# BV_2 = np.array([-4.663479471579194, -4.660001898184419, -5.002782924100757, -3.9846578147262335, -2.1544390972703695])
# ABV_Simple_2 = np.array([-0.26183422678150237, -0.3264209399931133, -0.21153253922238946, -0.22937510488554835, -0.249622508068569])
# ABV_Hessian_2 = np.array([-3.4270955465035513, 4.110725247301161, -2.75443596765399, -2.586051090620458, -14.515810911543667])

# SV_3 = np.array([-19.25022685329119, -21.277523232499757, -21.546828908721608, -23.08577266136805, -20.40882281263669]) 
# BV_3 = np.array([-3.404400574043393, -4.724271027371287, -4.565289748832583, -4.435775639489293, -3.7755289170891047])
# ABV_Simple_3 = np.array([-0.36896685021929443, -0.33507801103405654, -0.3182834757026285, -0.2369527635164559, -0.3191174070816487])
# ABV_Hessian_3 = np.array([-31.687707112170756, 1.1572337541729212, 22.60343938320875, -15.072019539773464, 9.443318340927362])

# NOISY:


def remove_outliers(*arrays, method="iqr", threshold=1.5):
    # stack the arrays into a single 2D array for synchronized processing
    data = np.column_stack(arrays)

    if method == "iqr":
        # compute iqr for each column (dimension)
        Q1 = np.percentile(data, 25, axis=0)
        Q3 = np.percentile(data, 75, axis=0)
        IQR = Q3 - Q1

        # define bounds for outliers
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # create a mask for non-outlier rows
        mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)

    elif method == "zscore":
        # compute Z-scores for each column (dimension)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        z_scores = (data - mean) / std

        # create a mask for non-outlier rows
        mask = np.all(np.abs(z_scores) <= threshold, axis=1)

    else:
        raise ValueError("Unsupported method. Use 'iqr' or 'zscore'.")

    # apply the mask to filter out rows with outliers
    filtered_data = data[mask]

    # return synchronized arrays
    return tuple(filtered_data[:, i] for i in range(filtered_data.shape[1]))

# Unpack the cleaned arrays
# SV_1, BV_1, ABV_Simple_1, ABV_Hessian_1 = remove_outliers(SV_1, BV_1, ABV_Simple_1, ABV_Hessian_1, method="iqr", threshold=1)
# SV_2, BV_2, ABV_Simple_2, ABV_Hessian_2 = remove_outliers(SV_2, BV_2, ABV_Simple_2, ABV_Hessian_2, method="iqr", threshold=1)
# SV_3, BV_3, ABV_Simple_3, ABV_Hessian_3 = remove_outliers(SV_3, BV_3, ABV_Simple_3, ABV_Hessian_3, method="iqr", threshold=1)



# ABV_Hessian_exp, SV_all = remove_outliers(ABV_Hessian_exp, SV_all, method="iqr", threshold=2)

# print(pearsonr(SV_all, ABV_Hessian_exp)[0])

from scipy.stats import boxcox

# Shift data to positive range if needed
ABV_Hessian_shifted = ABV_Hessian_all - np.min(ABV_Hessian_all) + 1  # Ensure all values > 0

# Apply Box-Cox transformation
ABV_Hessian_boxcox, lambda_ = boxcox(ABV_Hessian_shifted)

print(f"Optimal Box-Cox lambda: {lambda_}")
plt.scatter(SV_all, ABV_Hessian_boxcox)
plt.title('SV vs. ABV_Hessian (Box-Cox Transformed)')
plt.xlabel('SV')
plt.ylabel('Box-Cox(ABV_Hessian)')
plt.show()
print(pearsonr(SV_all, ABV_Hessian_boxcox)[0])
 
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = None

SV = []
BV = []
ABV_Simple = []
ABV_Hessian = []
for i in [1, 2, 3]:
    if scaler is not None:
        SV.extend(scaler.fit_transform(eval(f'SV_{i}').reshape(-1, 1)).flatten())
        BV.extend(scaler.fit_transform(eval(f'BV_{i}').reshape(-1, 1)).flatten())
        ABV_Simple.extend(scaler.fit_transform(eval(f'ABV_Simple_{i}').reshape(-1, 1)).flatten())
        ABV_Hessian.extend(scaler.fit_transform(eval(f'ABV_Hessian_{i}').reshape(-1, 1)).flatten())
    else:
        SV.extend(eval(f'SV_{i}'))
        BV.extend(eval(f'BV_{i}'))
        ABV_Simple.extend(eval(f'ABV_Simple_{i}'))
        ABV_Hessian.extend(eval(f'ABV_Hessian_{i}'))

# do a plot of sv and abv ABV_Simple
def transform(vector):
    # return [np.log(abs(x)+1e-10) for x in vector]
    return vector

def negate(vector):
    return [-x for x in vector]


SV_norm = transform(SV)
BV_norm = transform(BV)
ABV_Simple_norm = transform(ABV_Simple)
ABV_Hessian_norm = transform(ABV_Hessian)

# create df
data = {
    'SV_norm': SV_norm,
    'BV_norm': BV_norm,
    'ABV_Simple_norm': ABV_Simple_norm,
    'ABV_Hessian_norm': ABV_Hessian_norm
}
df = pd.DataFrame(data)

# calculate and display pcc for each pair
pairs = [
    ('SV_norm', 'BV_norm'),
    ('SV_norm', 'ABV_Simple_norm'),
    ('SV_norm', 'ABV_Hessian_norm'),
    # ('BV_norm', 'ABV_Simple_norm'),
    # ('BV_norm', 'ABV_Hessian_norm'),
    # ('ABV_Simple_norm', 'ABV_Hessian_norm')
]
for pair in pairs:
    x = df[pair[0]]
    y = df[pair[1]]
    pcc, p_value = pearsonr(x, y)
    print(f"PCC between {pair[0]} and {pair[1]}: {pcc:.8f} with p-value: {p_value:.8f}")