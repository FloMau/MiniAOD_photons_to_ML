import pandas as pd
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('datafile')
parser.add_argument('preselectionfile')
args = parser.parse_args()
df = pd.read_pickle(args.datafile)
preselection = np.load(args.preselectionfile)

### redo preselection from mymodules as check
pt = df['pt'] > 25  # no leading photon, because I do single photon studies
transition = (1.44 < df['eta'].abs()) & (df['eta'].abs() < 1.57)
eta = (df['eta'].abs() < 2.5) & (~transition)
HoE = df['HoE'] < 0.08
iso_gamma = df['I_gamma'] < 4.0
iso_track = df['I_tr'] < 6.0

# barrel
barrel = df['detID'] == 1
R9_small_barrel = df['r9'] > 0.5
R9_large_barrel = df['r9'] > 0.85
sigma_barrel = df['sigma_ieie'] < 0.015
barrel1 = barrel & R9_small_barrel & sigma_barrel & iso_gamma & iso_track
barrel2 = barrel & R9_large_barrel
barrel = barrel1 | barrel2

# endcap
endcap = df['detID'] == 0
R9_small_endcap = df['r9'] > 0.80
R9_large_endcap = df['r9'] > 0.90
sigma_endcap = df['sigma_ieie'] < 0.035
endcap1 = endcap & R9_small_endcap & sigma_endcap & iso_gamma & iso_track
endcap2 = endcap & R9_large_endcap
endcap = endcap1 | endcap2

# combine Masks
shower_shape = HoE & (barrel | endcap)
one_of = (df['r9'] > 0.8) | ((df['I_ch'] / df['pt']) < 0.3) | (df['I_ch'] < 20)
total_mask = pt & eta & shower_shape & one_of

#################################################################
def survive(filter):
    return df.real[filter].sum()/df.real.sum()

def reject(filter):
    return (~df.real[filter]).sum()/(~df.real).sum()

sel = total_mask
evetosel = sel*df.eveto

print('total photons:', len(df)/1e6, 'Mio')
print(f'total real photons: {df.real.sum()/1e6: .3f} Mio / {df.real.sum()/len(df)*100: .02f}%')
print(f'total fake photons: {(~df.real).sum()/1e6: .3f} Mio / {(~df.real).sum()/len(df)*100: .02f}%')
print(f'after preselection: {(df[sel].real).sum()/1e6: .3f} Mio / {(~df[sel].real).sum()/1e6: .3f} Mio '
      f'= {(df[sel].real).sum()/(~df[sel].real).sum(): .0f}:1')
print(f'with eveto: {(df[evetosel].real).sum()/1e6: .3f} Mio / {(~df[evetosel].real).sum()/1e6: .3f} Mio '
      f'= {(df[evetosel].real).sum()/(~df[evetosel].real).sum(): .0f}:1')
print()
print(f'real after eveto: {survive(df.eveto)*100:.02f}%')
print(f'real after pt-cut: {survive(pt)*100:.02f}%')
print(f'real after eta-cut: {survive(eta)*100:.02f}%')
print(f'real after pt- & eta-cut: {survive(pt & eta)*100:.02f}%')
print(f'real after all cuts: {survive(sel)*100:.02f}%')
print(f'real after all cuts and eveto: {survive(evetosel)*100:.02f}%')
print()
print(f'fake after eveto: {reject(df.eveto)*100:.02f}%')
print(f'fake after pt-cut: {reject(pt)*100:.02f}%')
print(f'fake after eta-cut: {reject(eta)*100:.02f}%')
print(f'fake after pt- & eta-cut: {reject(pt & eta)*100:.02f}%')
print(f'fake after all cuts: {reject(sel)*100:.02f}%')
print(f'fake after all cuts and eveto: {reject(evetosel)*100:.02f}%')








