# Readme



## Files

* main.py    **main code** to train, test and visualize 
* model.py    MultiDAE, MultiVAE, **DisenVAE**(MacridVAE), **DisenEVAE**(SEM-MacridVAE)
* data.py    load dataset and split into train, val_tr, val_te, test_tr, test_te
* RecomData    
  * ml-latest-small    sample dataset
    * prep.py    pre-process data
    * ratings.txt    rating actions file
    * embed.npy    image features file
  * prep.py    further process data
* run    save log and model



## Run

```bash
### part 1
./DecomData/<dataset>

### part 2
./DecomData
python prep.py <dataset>

### part 3
./
python main.py --data <dataset> --model <model> --device <cuda>


### Take ml-latest-small dataset as an example: 
./DecomData/ml-latest-small:  python prep.py
./DecomData:                  python prep.py ml-latest-small
./:                           python main.py --data ml-latest-small --model DisenEVAE --device cuda:0
```



