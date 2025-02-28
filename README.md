
# Source code for ML-ERC

## command
  
  ```
    python multi_train_pseudo.py --dataset [MELD/EMORY/iemocap] --batch 16 --loss [bce/weighted/supcon/SCL/SLCL/JSCL/JSPCL/ICL]
  ```

## Key arguments

- ```dataset```: [MELD / EMORY / iemocap]
- ```pseudo```: Whether to use pseudo labeing [True / False]
- ```entropy```: Whether to use entropy weight in WSCL [True / False]
- ```multi_weighted```: Whether to use label relation weight in WSCL [True / False]
- ```alpha```: the parameter to adjust values between BCE loss and multi-CL loss
- ```beta``` : the parameter to adjust values between CE loss and ML-ERC loss
- ```pseudo_entropy``` : gamma, the threshold for pseudo labeling
 
## Requirements
- torch 1.7.1
- torchtext 0.8.1
- pytorch-crf 0.7.2
- scikit-learn 0.23.1


