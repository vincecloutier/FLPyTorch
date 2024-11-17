import numpy as np
from scipy.stats import pearsonr
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

#IID
# SV_1 = np.array([-24.409884888927145, -22.62767086327076, -23.909745270013808, -22.760746810833616, -23.29781094590823]) 
# BV_1  = np.array([-5.948816725984216, -5.545571552589536, -5.311726374551654, -5.041963508352637, -5.593782817944884])
# ABV_Simple_1 = np.array([-0.252875097328797, -0.2288856585510075, -0.2835275926627219, -0.2190842186100781, -0.2911171605810523])
# ABV_Hessian_1 = np.array([-2.5707323076203465, -5.749352287501097, -22.565373002551496, 17.26112177222967, 18.256525065749884])

# SV_2 = np.array([-23.173703097303715, -25.61744375824929, -24.713261716564496, -22.644227696458497, -24.378510726491612])
# BV_2 = np.array([-4.706250285729766, -6.083594957366586, -5.773475898429751, -5.325923508033156, -6.170815190300345])
# ABV_Simple_2 = np.array([-0.3507651600521058, -0.32443248899653554, -0.2853101100772619, -0.312060855794698, -0.3725827857851982])
# ABV_Hessian_2 = np.array([-6.704698405228555, -9.542349236086011, 8.566787237301469, 13.532176872715354, 3.014599473681301])


# NON IID:
SV_1 = np.array([181.95880798796816, -38.19942807853222, -75.77778747876486, -35.91375678976377, -64.00016624927521])
BV_1 = np.array([78.77293125167489, -21.987386893481016, -47.43576893955469, -27.594971112906933, -37.032924480736256])
ABV_Simple_1 = np.array([1.141512457281351, -0.5619645267724991, -0.9291043318808079, -0.9759988049045205, -0.0395272308960557])
ABV_Hessian_1 = np.array([643.036488163285, -80.68008410930634, -183.78021064121276, -141.8907394418493, -1.2421450885012746])

SV_2 = np.array([-47.21579605738322, 156.01760527094203, -41.66360622048378, -65.57308938503266, -55.69278532663981])
BV_2 = np.array([-26.833819838240743, 66.41561008058488, -19.051439123228192, -38.669827895238996, -28.168856808915734])
ABV_Simple_2 = np.array([-0.9342706445604563, 1.087217248044908, -0.28548065200448036, -0.3898796336725354, -0.7880192287266254])
ABV_Hessian_2 = np.array([-240.99354546517134, 573.586022160016, -39.10542896948755, -25.002182245254517, -98.89350616931915])

SV_3 = np.array([-45.93091857532661, -74.23531784216563, 145.5069699873527, -75.99037101964154, -20.155097257097562])
BV_3 = np.array([-20.130715768784285, -41.4580530077219, 68.27663829550147, -39.45971083268523, -9.218875374644995])
ABV_Simple_3 = np.array([-0.8013164550065994, -0.7688949685543776, 1.17510259244591, -1.0703009562566876, -1.4710255786776543])
ABV_Hessian_3 = np.array([-138.41716212034225, -75.8597601056099, 1094.956063770689, -187.7881381334737, -502.75566007755697])


# MISLABELED:


# NOISY:


scaler = MinMaxScaler()
# scaler = None

SV = []
BV = []
ABV_Simple = []
ABV_Hessian = []
for i in range(1, 4):
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

print(SV)
print(BV)
print(ABV_Simple)
print(ABV_Hessian)

# do a plot of sv and abv ABV_Simple
def transform(vector):
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