import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import boost_histogram as bh

import myplotparams
from mymodules import plot_subset

df = pd.read_pickle('data/combined.pkl')
# don't need preselections

r9 = df['r9']
sigma = df['sigma_ieie']
converted = df['converted'] | df['convertedOneLeg']
barrel = df['detID']
endcap = ~barrel
real = df['real']
fake = ~real




################################
fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
ax1, ax2, ax3, ax4 = axs.flat
plot_subset(ax1, r9[real & barrel], converted, (30, 0.2, 1.11), colors=['black', 'red', 'blue'], labels='all, converted, unconverted'.split(', '))
plot_subset(ax2, r9[real & endcap], converted, (30, 0.2, 1.11), colors=['black', 'red', 'blue'], labels='all, converted, unconverted'.split(', '))
plot_subset(ax3, r9[fake & barrel], converted, (30, 0.2, 1.11), colors=['black', 'red', 'blue'], labels='all, converted, unconverted'.split(', '))
plot_subset(ax4, r9[fake & endcap], converted, (30, 0.2, 1.11), colors=['black', 'red', 'blue'], labels='all, converted, unconverted'.split(', '))

ax1.set_title('Barrel')
ax2.set_title('Endcap')
ax1.set_ylabel('real')
ax3.set_ylabel('fake')
for ax in axs.flat:
    ax.set_yscale('log')
    ax.set_xlabel('$R_9$')
    ax.legend()

fig.savefig('plots/r9_conversions_log.pdf')
fig.savefig('plots/r9_conversions_log.png')
############################
fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
ax1, ax2, ax3, ax4 = axs.flat
plot_subset(ax1, r9[real & barrel], converted, (30, 0.2, 1.11), colors=['black', 'red', 'blue'], labels='all, converted, unconverted'.split(', '))
plot_subset(ax2, r9[real & endcap], converted, (30, 0.2, 1.11), colors=['black', 'red', 'blue'], labels='all, converted, unconverted'.split(', '))
plot_subset(ax3, r9[fake & barrel], converted, (30, 0.2, 1.11), colors=['black', 'red', 'blue'], labels='all, converted, unconverted'.split(', '))
plot_subset(ax4, r9[fake & endcap], converted, (30, 0.2, 1.11), colors=['black', 'red', 'blue'], labels='all, converted, unconverted'.split(', '))

ax1.set_title('Barrel')
ax2.set_title('Endcap')
ax1.set_ylabel('real')
ax3.set_ylabel('fake')
for ax in axs.flat:
    ax.set_ylim(0, 0.33)
    ax.set_xlabel('$R_9$')
    ax.legend()

fig.savefig('plots/r9_conversions.pdf')
fig.savefig('plots/r9_conversions.png')

################################# sigma
fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
ax1, ax2, ax3, ax4 = axs.flat
plot_subset(ax1, sigma[real & barrel], converted, (30, 0, 0.05), colors=['black', 'red', 'blue'], labels='all, converted, unconverted'.split(', '))
plot_subset(ax2, sigma[real & endcap], converted, (30, 0, 0.05), colors=['black', 'red', 'blue'], labels='all, converted, unconverted'.split(', '))
plot_subset(ax3, sigma[fake & barrel], converted, (30, 0, 0.05), colors=['black', 'red', 'blue'], labels='all, converted, unconverted'.split(', '))
plot_subset(ax4, sigma[fake & endcap], converted, (30, 0, 0.05), colors=['black', 'red', 'blue'], labels='all, converted, unconverted'.split(', '))

ax1.set_title('Barrel')
ax2.set_title('Endcap')
ax1.set_ylabel('real')
ax3.set_ylabel('fake')
for ax in axs.flat:
    ax.set_ylim(0, 0.8)
    ax.set_xlabel('$\sigma_{ieie}$')
    ax.legend()

fig.savefig('plots/sieie_conversions.pdf')
fig.savefig('plots/sieie_conversions.png')

################################
fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
ax1, ax2, ax3, ax4 = axs.flat
plot_subset(ax1, sigma[real & barrel], converted, (30, 0, 0.05), colors=['black', 'red', 'blue'], labels='all, converted, unconverted'.split(', '))
plot_subset(ax2, sigma[real & endcap], converted, (30, 0, 0.05), colors=['black', 'red', 'blue'], labels='all, converted, unconverted'.split(', '))
plot_subset(ax3, sigma[fake & barrel], converted, (30, 0, 0.05), colors=['black', 'red', 'blue'], labels='all, converted, unconverted'.split(', '))
plot_subset(ax4, sigma[fake & endcap], converted, (30, 0, 0.05), colors=['black', 'red', 'blue'], labels='all, converted, unconverted'.split(', '))

ax1.set_title('Barrel')
ax2.set_title('Endcap')
ax1.set_ylabel('real')
ax3.set_ylabel('fake')
for ax in axs.flat:
    ax.set_yscale('log')
    ax.set_xlabel('$\sigma_{ieie}$')
    ax.legend()

fig.savefig('plots/sieie_conversions_log.pdf')
fig.savefig('plots/sieie_conversions_log.png')

plt.show()

