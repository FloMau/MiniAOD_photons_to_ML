import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import myplotparams
from mymodules import histplot

rechits = np.load('data/rechits/all_rechits.npy')
rechits = rechits[rechits > 0].flatten()

rechits = np.log(rechits)

fig, ax = plt.subplots()

bins = (100, -6, 3.5)

histplot(ax, rechits, bins, normalizer=None)
# ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('ln(E) [GeV]')
ax.set_ylabel('#')
ax.set_title('rechits energy distribution')

plt.tight_layout()


plt.savefig('rechitsdist.png')
print('plot saved as: rechitsdist.png')
print('FINISHED')




