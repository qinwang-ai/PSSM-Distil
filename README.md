# PSSM-Distil
The paper "PSSM-Distil: Protein Secondary Structure Prediction (PSSP) on Low-QualityPSSM by Knowledge Distillation with Contrastive Learning" under review of AAAI Conference

> Requirement
>
>
    pip install torch
    pip install glob
    pip install tqdm
    pip install numpy

> Instructions

    python inference_real.py
    
Aboving commend will predict secondary 
structure of sequence in 'sequences' folder with pssm in 'low_pssm' folder and print the accuracy.

    python inference_our.py
    
Aboving commend will predict secondary structure of sequence in 'sequences' folder with enhanced pssm which refined by PSSM-Distil and print the accuracy. 
Besides, this commend will save a enhanced pssm file in 'enhanced_pssms' folder.

    python sample_msa_from_pssm.py ./low_pssms/6dnqE.npy

Aboving commend will sample MSAs from a specifical PSSM file. Then 2000 MSAs will save in 'a3ms' folder.

    python sample_msa_from_pssm.py ./enhanced_pssms/6dnqE.npy
    
Aboving commend will sample 2000 MSAs from enhanced PSSM file and save in a3ms folder.

> Visualization

Please upload two MSA files in 'a3ms' folder to the website: https://weblogo.berkeley.edu/logo.cgi respectively.
Then you will see such comparison images.

<img src="./img/file5ca7md.png" width="500px" alt='low real PSSM'/>
<img src="./img/filelXoHzj.png" width="500px" alt='enhanced PSSM'/>

> BC40 dataset

https://bit.ly/35mC3Mx

The dataset we released to examine the performance of PSSM-Distil


