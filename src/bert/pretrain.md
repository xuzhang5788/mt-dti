## KIBA
### smiles
* n: 2111
* max: 590
* n_{>100}: 57

* 590: 0:99, 100:199, 200:299, 300:399, 400:499, 500:590 (mean of 6 vectors)

## Training Time

* n: 97M
* steps: 10000000
* batches: 32
* examples: 10000000*32 = 320,000,000 = 320M
* avg n of repeat each sample: 320/97~3
* V100: 100samples/1sec
    * 10000000/100/60/60~28h
* TPU?
