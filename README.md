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
    
Aboving commend will sample MSAs from a specifical PSSM file. Then 2000 MSAs will save in 'a3ms' folder.

    python sample_msa_from_pssm.py ./enhanced_pssms/4ynhA.npy
    
Aboving commend will sample 2000 MSAs from enhanced PSSM and save in a3ms folder as '4ynhA_enhanced_pssms.a3m'.

> Visualization

Please upload original low-quality '.a3m' file in 'a3ms' folder and enhanced one in 'a3ms' folder to the website: https://weblogo.berkeley.edu/logo.cgi respectively.
Then you will see such comparison images.

<img src="./img/file5ca7md.png" width="500px" alt='low real PSSM'/>
<img src="./img/filelXoHzj.png" width="500px" alt='enhanced PSSM'/>

> BC40 dataset

https://bit.ly/35mC3Mx

The dataset we released to examine the performance of PSSM-Distil


