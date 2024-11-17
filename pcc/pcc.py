import numpy as np
from scipy.stats import pearsonr
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

#IID
SV_1 = np.array([18.58035978, 18.38290713, 17.79816145, 19.05175414, 19.08264337]) 
BV_1  = np.array([3.36727499, 3.27869586, 2.81406335, 3.56837929, 3.49285300])
ABV_Simple_1 = np.array([-0.13069274, -0.09832043, -0.09446989, -0.10965246, -0.12688119])
ABV_Hessian_1 = np.array([-11.54136181, -2.18881949, -4.48050083, 22.10438061, -1.91315184])

# SV_2  = np.array([18.736357309420914, 18.90407821337382, 18.416281603773438, 19.144453667600946, 18.53265393575033])
# BV_2 = np.array([3.9011393524706364, 3.802803013473749, 3.377421498298645, 3.736703708767891, 3.541977170854807])
# ABV_Simple_2 = np.array([-0.12329316232353449, -0.16612789873033762, -0.2394404369406402, -0.20201075123623013, -0.15620961552485824])
# ABV_Hessian_2 = np.array([2.2202022494748235, 13.033299993723631, -9.206642349250615, -2.9683663360774517, -4.447527632117271])


# SV_3 = np.array([20.460480530063304, 19.66832312345505, 18.92978727718194, 18.877563068270685, 19.808013507723807])
# BV_3 = np.array([3.846090327948332, 3.682807870209217, 3.2508517764508724, 3.4746198691427708, 3.794637631624937])
# ABV_Simple_3 = np.array([-0.07564560556784272, -0.21023885486647487, -0.23076834809035063, -0.1678106877952814, -0.1974808075465262])
# ABV_Hessian_3 = np.array([-4.18807914853096, -13.836310371756554, 6.702124995179474, -10.304717208258808, 1.1932640168815851])


SV_2 = np.array([23.03181714514891, 24.446331180135413, 21.988261254628505, 22.953515303134925, 21.530984731515254])
BV_2 = np.array([5.709950456395745, 6.176261218264699, 4.982393695041537, 5.507119355723262, 4.385137466713786])
ABV_Simple_2 = np.array([-0.20504500064998865, -0.16275964071974158, -0.21334836073219776, -0.2175194383598864, -0.19968532864004374])
ABV_Hessian_2 = np.array([-5.302127815783024, -2.725527008995414, -0.9665978103876114, -3.6171667221933603, 2.0389154190197587])

SV_3 = np.array([22.618458191553756, 21.336152354876198, 21.83671866854032, 21.824517483512555, 21.847085768977795])
BV_3 = np.array([5.23212487436831, 5.26935407705605, 4.723152415826917, 5.10902095399797, 5.409222355112433])
ABV_Simple_3 = np.array([-0.20623072097077966, -0.2394038075581193, -0.18684212490916252, -0.1704110074788332, -0.17517958860844374])
ABV_Hessian_3 = np.array([5.409855573438108, -5.669278301298618, 0.8060918913688511, -2.1735211610794067, -1.3122559823095798])

# # NON IID:
# SV_1 = np.array([31.291860879460973, 32.84774865309397, -138.7057807981968, 68.60380674004556, 81.22258374194305])
# BV_1 = np.array([29.193171689286828, 11.661331435665488, -60.56910111196339, 35.4456782322377, 40.883232997730374])
# ABV_Simple_1 = np.array([-0.5455400366336107, -0.35817775037139654, 0.6985825952142477, -0.7804434234276414, -0.612911269068718])
# ABV_Hessian_1 = np.array([-151.50582993216813, -16.52216829266399, 513.473242521286, -76.14129175338894, -78.48669195175171])

# SV_2 = np.array([-151.92581943074862, 23.970193034410475, 64.22025146186351, 66.20461304088434, 47.120693977673845])
# BV_2 = np.array([-68.84151697531343, 7.729100879281759, 35.12318518012762, 34.71625126898289, 21.705102276057005])
# ABV_Simple_2 = np.array([0.7356966622173786, -0.1753024347126484, -0.8127893451601267, -0.2609550068154931, -0.6995107722468674])
# ABV_Hessian_2 = np.array([505.4269308820367, -52.55351094156504, -205.8861021772027, -0.8249151716008782, -83.75985838891938])

# SV_3 = np.array([-132.25694383978848, 44.7145905315876, 43.11766633590063, 19.62747511466344, 80.19554695685704])
# BV_3 = np.array([-57.348339738324285, 23.391695214435458, 11.374541876837611, 16.740133782848716, 39.12223184667528])
# ABV_Simple_3 = np.array([0.9266111422330141, -1.6125200507231057, -1.6915232394821942, -0.49343179422430694, -1.0040365098975599])
# ABV_Hessian_3 = np.array([1377.8495567962527, -385.17626748932526, -568.8530850410461, -124.0806631508749, -291.70372668048367])


# scaler = MinMaxScaler()
scaler = None

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
    return [np.log(abs(x) + 1e-10) for x in vector]


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